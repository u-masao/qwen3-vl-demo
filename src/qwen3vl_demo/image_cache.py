"""生成画像のローカルファイルシステムキャッシュ。

画像生成（diffusers）は ``0.2 枚/秒`` 程度と遅く、同じモデル・同じ入力なら同じ画像が出る
（生成を決定的にしている。:func:`derive_seed` 参照）。そこで生成済み画像をローカルに
キャッシュし、ヒット時は生成（およびモデルロード）をまるごとスキップする。

設計方針
--------
* **バックエンドは素のファイルシステム**（1 画像 1 PNG）。実測 0.2 枚/秒では I/O は完全に
  誤差なので、SQLite や Redis のような I/O 最適化は不要。依存ゼロ・最小実装を優先する。
* キャッシュは**環境ごとの使い捨て**。既定の保存先は ``.cache/imggen``（``.gitignore`` の
  ``.cache/`` で除外済み）で、Git 管理も DVC 管理もしない。消して作り直せる前提。
* キーは「出力画像を一意に決める入力すべて」の SHA-256。入力が 1 つでも変われば別キーになる。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# キャッシュのエンコード仕様バージョン。PNG 化や seed 導出など「同じ入力でも出力バイト列が
# 変わる」変更を入れたら上げて、古いキャッシュを一括で無効化する。
CACHE_VERSION = "v1"


def derive_seed(base_seed: int, prompt: str) -> int:
    """``base_seed`` とプロンプトから決定的な per-image シードを導く。

    画像をプロンプト単位で決定的にするための要。これにより生成順・バッチサイズ・件数に
    依存せず「同じ base_seed・同じプロンプトなら同じ画像」が保証され、プロンプトをキーに
    したキャッシュが健全になる（diffusers にはプロンプトごとの Generator リストを渡す）。

    torch の ``manual_seed`` が安全に受け取れるよう 63bit 非負整数に丸める。
    """
    digest = hashlib.sha256(f"{base_seed}:{prompt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class ImageCache:
    """生成画像のファイルシステムキャッシュ（``<root>/<key[:2]>/<key>.png``）。"""

    def __init__(self, root: Path | str, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def key(
        self,
        *,
        model_id: str,
        prompt: str,
        seed: int,
        steps: int,
        guidance: float,
        size: int,
        dtype: str,
    ) -> str:
        """出力画像を一意に決める入力すべてから安定なキー（SHA-256 hex）を作る。"""
        payload = {
            "version": CACHE_VERSION,
            "model_id": model_id,
            "prompt": prompt,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "size": size,
            "dtype": dtype,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        # 設定違いでキャッシュが大量になっても 1 ディレクトリに集中しないよう 2 階層に分散。
        return self.root / key[:2] / f"{key}.png"

    def get(self, key: str) -> Image.Image | None:
        """ヒットすれば PIL 画像を返し、ミス（または無効時）は ``None`` を返す。"""
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        with Image.open(path) as img:
            return img.convert("RGB")

    def put(self, key: str, image: Image.Image) -> None:
        """画像を PNG で atomic に保存する（無効時は no-op）。"""
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 同じディレクトリの一時ファイルに書いてから rename。途中クラッシュでも壊れた
        # PNG を残さない（os.replace は同一 FS なら atomic）。
        tmp = path.with_suffix(f".png.tmp.{os.getpid()}")
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
