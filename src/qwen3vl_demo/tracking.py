"""MLflow による評価・学習の実験管理ヘルパ（Issue #9）。

評価（``evaluate.py``）とリランク（``rerank.py``）の結果を同一 Experiment ``"evaluate"`` に、
学習（``train.py`` / ``train_reranker.py``）を Experiment ``"train"`` に run として記録する。
これにより Retriever（``rerank=none``）と Reranker が同じ Experiment 内の比較可能な run として
並び、MLflow UI で精度・速度・メモリを横断比較できる。

記録する内容（精度・処理速度・メモリ消費に興味があるため）:

  * **params** … 起動引数（``args.*``）＋ 解決後の全設定（``cfg.*``）を漏れなく記録する。
  * **metrics** … 評価器の返り dict（NDCG / Recall@k / MRR 等）を丸ごと記録する。
  * **学習曲線** … 学習中の loss / eval 指標を step 付きで記録する（:func:`make_curve_callback`）。
  * **所要時間** … 各工程の経過時間を ``time.*`` メトリクスで記録する（:class:`Timer` / :func:`log_time`）。
  * **System Metrics** … MLflow の system metrics で CPU / メモリ / GPU 使用率・VRAM の
    時系列を自動収集する（:func:`enable_system_metrics`。``psutil`` / ``pynvml`` が必要）。

Tracking バックエンドは **SQLite3**（``sqlite:///<repo>/mlflow.db``）を既定とする。
artifact は DB とは別管理になるため、置き場を ``<repo>/mlflow/<exp_id>/`` という既定風の
レイアウトに固定する（実行ディレクトリ依存で散らばらないようにするため。現状 artifact は
出力していないが将来用）。``MLFLOW_TRACKING_URI`` が設定されていればそちら（と接続先の既定
artifact 置き場）を優先する（サーバ運用への差し替え）。mlflow が使えない場合は警告して
no-op で通し、本体は止めない。
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import re
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

logger = logging.getLogger(__name__)

# 評価系 / 学習系 / データ生成の Experiment 名。
EXPERIMENT_NAME = "evaluate"
TRAIN_EXPERIMENT_NAME = "train"
DATA_EXPERIMENT_NAME = "generate_data"

# MLflow のキーで許可されない文字（英数・``_`` ``-`` ``.`` ``/`` 空白 ``:`` 以外）を ``_`` に置換する。
# 例: ``ndcg@10`` の ``@`` は不許可なので ``ndcg_at_10`` にしてから残りをサニタイズする。
_INVALID_KEY = re.compile(r"[^\w\-./ :]")

# system metrics を二重に有効化しないためのフラグ（プロセス内で 1 回だけ有効化する）。
_SYSTEM_METRICS_ENABLED = False


def args_to_params(args: argparse.Namespace) -> dict[str, Any]:
    """argparse 引数を ``args.*`` プレフィックス付きの params dict にする。

    未指定のオーバーライド引数（``config._UNSET`` センチネル）は「起動時に渡していない」
    ので除外する。``--seed 42`` のように実際に渡した値（と通常の既定値）だけが残る。
    """
    from .config import _UNSET

    return {f"args.{k}": v for k, v in vars(args).items() if v is not _UNSET}


def config_to_params(cfg: Any) -> dict[str, Any]:
    """解決後の Config を ``cfg.*`` のドット区切り params に平坦化する。

    起動引数（``args.*``）が「何を渡したか」なのに対し、こちらは「実際に使われた全設定値」を
    漏れなく残すためのもの。ネストした dataclass はドットで連結する（例: ``cfg.embedding.model_id``）。
    """
    if not is_dataclass(cfg):
        return {}
    out: dict[str, Any] = {}

    def _flatten(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _flatten(f"{prefix}.{k}", v)
        else:
            out[prefix] = value

    _flatten("cfg", asdict(cfg))
    return out


def _sanitize_key(key: str) -> str:
    """MLflow のメトリクスキー制約に合わせてキー名をサニタイズする。"""
    return _INVALID_KEY.sub("_", key.replace("@", "_at_"))


def _configure_tracking_uri(mlflow) -> None:  # noqa: ANN001
    """MLflow のバックエンドを SQLite3 に設定する（``MLFLOW_TRACKING_URI`` 未設定時のみ）。

    DB はリポジトリルートの ``mlflow.db``。``mlflow ui --backend-store-uri sqlite:///mlflow.db``
    で閲覧できる。環境変数が設定済みなら尊重する（サーバ運用などへの差し替え用）。
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    from .config import REPO_ROOT

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")


def _ensure_experiment(mlflow, name: str) -> None:  # noqa: ANN001
    """Experiment を選択する（無ければ作成）。

    SQLite バックエンドでは artifact は DB と別管理。MLflow 既定（``./mlruns``）ではなく、
    ``<repo>/mlflow/<exp_id>/`` という既定風レイアウトに置く。クライアント単体では既定の
    artifact root を差し替えられないため、次に割り振られる exp_id を予測して
    ``artifact_location`` に明示することで ``<exp_id>`` 階層を再現する。http(s) のリモート
    サーバ運用時はサーバ側の既定に任せる。
    """
    if mlflow.get_experiment_by_name(name) is not None:
        mlflow.set_experiment(name)
        return

    uri = mlflow.get_tracking_uri()
    if not uri.startswith(("http://", "https://")):
        from mlflow.entities import ViewType

        from .config import REPO_ROOT

        # 既存（削除済み含む）の最大 exp_id + 1 = 次に割り振られる id を予測する。
        existing = mlflow.search_experiments(view_type=ViewType.ALL)
        next_id = max((int(e.experiment_id) for e in existing), default=-1) + 1
        location = (REPO_ROOT / "mlflow" / str(next_id)).as_uri()
        with contextlib.suppress(Exception):  # 競合作成は set_experiment 側で吸収
            mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)


def enable_system_metrics(sampling_interval: float = 5.0) -> None:
    """MLflow の System Metrics（CPU / メモリ / GPU）収集を有効化する（プロセス内 1 回）。

    アクティブな run の実行中、別スレッドが一定間隔で CPU 使用率・メモリ・GPU 使用率・VRAM を
    サンプリングして記録する。``psutil``（CPU/メモリ）と ``pynvml``（NVIDIA GPU）が必要。
    """
    global _SYSTEM_METRICS_ENABLED
    if _SYSTEM_METRICS_ENABLED:
        return
    try:
        import mlflow

        mlflow.enable_system_metrics_logging()
        with contextlib.suppress(Exception):
            mlflow.set_system_metrics_sampling_interval(sampling_interval)
        _SYSTEM_METRICS_ENABLED = True
        logger.info("MLflow System Metrics を有効化しました（間隔 %.0fs）", sampling_interval)
    except Exception as exc:  # noqa: BLE001 - 収集できなくても本体は止めない
        logger.warning("MLflow System Metrics を有効化できません: %s", exc)


@contextlib.contextmanager
def start_run(
    run_name: str,
    *,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    experiment: str = EXPERIMENT_NAME,
    nested: bool = False,
) -> Iterator[None]:
    """MLflow run を開始するコンテキストマネージャ。

    ``nested=True`` のときは現在アクティブな run の子 run として開始する
    （Experiment は親に従う）。mlflow が使えない場合は警告して no-op で通す。
    """
    try:
        import mlflow
    except Exception as exc:  # noqa: BLE001 - mlflow 未導入でも本体は止めない
        logger.warning("mlflow が使えないため実験記録をスキップします: %s", exc)
        yield
        return

    if not nested:
        _configure_tracking_uri(mlflow)
        _ensure_experiment(mlflow, experiment)
    with mlflow.start_run(run_name=run_name, nested=nested):
        if tags:
            mlflow.set_tags(dict(tags))
        if params:
            mlflow.log_params(dict(params))
        yield


@contextlib.contextmanager
def cli_run(
    experiment: str,
    run_name: str,
    *,
    args: argparse.Namespace,
    cfg: Any,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """CLI 1 実行 = 1 run。**引数解決直後に開き、CLI 終了直前に閉じる**ためのラッパ。

    各 ``main()`` が worker 呼び出し全体をこれで囲むことで、run の実行ウィンドウが
    モデルロード・データ準備・本処理まで CLI 全体を覆う（System Metrics と所要時間が
    全工程をカバーする）。起動引数（``args.*``）と解決後の全設定（``cfg.*``）を params に
    記録し、System Metrics を有効化する。
    """
    enable_system_metrics()
    params = {**args_to_params(args), **config_to_params(cfg)}
    with start_run(run_name=run_name, params=params, tags=tags, experiment=experiment):
        yield


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    """数値メトリクスを現在の run に記録する（キーをサニタイズ、数値以外は無視）。"""
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        clean = {
            _sanitize_key(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        if clean:
            mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # noqa: BLE001 - 記録失敗で本体は止めない
        logger.warning("mlflow へのメトリクス記録に失敗しました: %s", exc)


def gpu_memory_status() -> dict[str, float] | None:
    """torch CUDA アロケータの予約/確保ピークと物理 VRAM を比べ、共有メモリ退避を検出する。

    WSL2 では確保が物理 VRAM を超えても NVIDIA driver が共有システムメモリへ退避して成功する
    （その分 PCIe 経由で激遅化する）。``nvidia-smi`` の使用量は物理上限で頭打ちになるため、
    退避の有無は **torch の予約量 ``memory_reserved`` / 確保量 ``max_memory_allocated``** を
    物理 VRAM ``total_memory`` と比較して判定するのが確実。CUDA 非対応なら None。

    返すキー（プロセス開始からのピーク。各 CLI は別プロセスなのでリセット不要）:
      * ``vram.total_mib`` … 物理 VRAM
      * ``vram.peak_reserved_mib`` … アロケータ予約ピーク
      * ``vram.peak_allocated_mib`` … テンソル確保ピーク（実working set）
      * ``vram.peak_spill_mib`` … max(0, 予約ピーク - 物理)＝共有メモリへ退避した量
      * ``vram.spilled`` … 退避していれば 1.0（予約ピークが物理を超過）
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        mib = 1024 * 1024
        total = torch.cuda.get_device_properties(0).total_memory / mib
        peak_reserved = torch.cuda.max_memory_reserved() / mib
        peak_alloc = torch.cuda.max_memory_allocated() / mib
        return {
            "vram.total_mib": total,
            "vram.peak_reserved_mib": peak_reserved,
            "vram.peak_allocated_mib": peak_alloc,
            "vram.peak_spill_mib": max(0.0, peak_reserved - total),
            "vram.spilled": 1.0 if peak_reserved > total else 0.0,
        }
    except Exception:  # noqa: BLE001 - torch 無し/CPU でも無視してよい
        return None


def log_gpu_memory_status() -> None:
    """:func:`gpu_memory_status` をアクティブ run に記録し、退避していれば警告ログを出す。"""
    status = gpu_memory_status()
    if status is None:
        return
    log_metrics(status)
    if status["vram.spilled"]:
        logger.warning(
            "[VRAM] 共有メモリへ退避: 予約ピーク %.0f MiB > 物理 %.0f MiB（+%.0f MiB 退避）。"
            "max_pixels / batch を下げて 16GB に収めることを推奨。",
            status["vram.peak_reserved_mib"],
            status["vram.total_mib"],
            status["vram.peak_spill_mib"],
        )


class Timer:
    """``with Timer() as t:`` で経過秒数を ``t.elapsed`` に記録するコンテキストマネージャ。"""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> bool:
        self.elapsed = time.perf_counter() - self._t0
        return False


@contextlib.contextmanager
def log_time(metric_name: str) -> Iterator[None]:
    """ブロックの所要時間を ``metric_name`` メトリクスとして現在の run に記録する。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log_metrics({metric_name: time.perf_counter() - t0})


def make_curve_callback():
    """学習曲線（loss / eval 指標）を step 付きで MLflow に記録する TrainerCallback を返す。

    transformers / sentence-transformers の Trainer に ``callbacks=[...]`` で渡す。
    メトリクスキーはサニタイズするため、``eval_..._ndcg@10`` のような ``@`` を含むキーでも
    安全に記録できる（組み込み MLflowCallback は ``@`` でエラーになりうる）。mlflow や
    transformers が無い場合は ``None`` を返す。
    """
    try:
        from transformers import TrainerCallback
    except Exception as exc:  # noqa: BLE001
        logger.warning("TrainerCallback が使えないため学習曲線の記録をスキップします: %s", exc)
        return None

    class _MlflowCurveCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs or not state.is_world_process_zero:
                return
            log_metrics(logs, step=state.global_step)

    return _MlflowCurveCallback()
