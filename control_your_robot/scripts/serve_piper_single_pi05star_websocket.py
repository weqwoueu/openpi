import argparse
import dataclasses
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "openpi-client" / "src"))

from openpi.policies import policy_config as _policy_config  # noqa: E402
from openpi.serving import websocket_policy_server  # noqa: E402
from openpi.training import config as _config  # noqa: E402

DEFAULT_TRAIN_CONFIG = "pi05_piperx_plug_sft"


def _validate_checkpoint_dir(checkpoint_dir: str, asset_id: str) -> Path:
    path = Path(checkpoint_dir).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    params_dir = path / "params"
    pytorch_weights = path / "model.safetensors"
    if not (params_dir.is_dir() or pytorch_weights.is_file()):
        raise ValueError(f"checkpoint must contain params/ or model.safetensors: {path}")
    if params_dir.is_dir() and not (params_dir / "_METADATA").is_file():
        raise ValueError(f"incomplete JAX checkpoint, params/_METADATA is missing: {path}")
    norm_stats = path / "assets" / asset_id / "norm_stats.json"
    if not norm_stats.is_file():
        raise ValueError(f"checkpoint norm stats not found: {norm_stats}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Piper 单臂 websocket 推理服务（默认 pi05，兼容 PiStar）")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="checkpoint 根目录，应包含 params/ 或 model.safetensors",
    )
    parser.add_argument("--train-config", type=str, default=DEFAULT_TRAIN_CONFIG, help="openpi 训练配置名")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--default-prompt", type=str, default=None, help="请求中缺省 prompt 时使用")
    parser.add_argument(
        "--adv-guidance-beta",
        type=float,
        default=None,
        help="PiStar CFG guidance scale；普通 pi0.5 留空",
    )
    args = parser.parse_args()

    train_config = _config.get_config(args.train_config)
    is_pistar = bool(getattr(getattr(train_config, "model", None), "pistar", False))
    if is_pistar:
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(train_config.data, adv_ind_dropout=False),
        )
    if args.adv_guidance_beta is not None and not is_pistar:
        parser.error("--adv-guidance-beta can only be used with a PiStar config")
    if args.adv_guidance_beta is not None and not math.isfinite(args.adv_guidance_beta):
        parser.error("--adv-guidance-beta must be finite")

    asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
    if not isinstance(asset_id, str) or not asset_id:
        parser.error(f"train config {args.train_config!r} does not define an asset id")
    checkpoint_dir = _validate_checkpoint_dir(args.checkpoint_dir, asset_id)
    sample_kwargs = None
    if args.adv_guidance_beta is not None:
        sample_kwargs = {"adv_guidance_beta": args.adv_guidance_beta}
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs=sample_kwargs,
        default_prompt=args.default_prompt,
    )

    effective_guidance_beta = None
    if is_pistar:
        effective_guidance_beta = (
            args.adv_guidance_beta
            if args.adv_guidance_beta is not None
            else getattr(train_config.model, "adv_guidance_beta", None)
        )

    metadata = dict(policy.metadata or {})
    metadata.update(
        {
            "deploy_mode": "pi05star" if is_pistar else "pi05",
            "train_config": args.train_config,
            "requires_adv_ind": is_pistar,
            "adv_guidance_beta": effective_guidance_beta,
        }
    )

    print("=" * 50)
    print("Piper 单臂 websocket 推理服务")
    print("=" * 50)
    print(f"checkpoint: {checkpoint_dir}")
    print(f"train_config: {args.train_config}")
    print(f"deploy_mode: {'pi05star' if is_pistar else 'pi05'}")
    if is_pistar:
        print(f"adv_guidance_beta: {effective_guidance_beta}")
    print(f"listen: ws://{args.host}:{args.port}")
    print("=" * 50)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl+C received, inference server stopped normally.")
