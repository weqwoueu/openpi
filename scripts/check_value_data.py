"""Quick sanity check for value dataset batches."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax
import numpy as np

from openpi.models.value_model_config import ValueModelConfig
from openpi.shared import console
import openpi.training.data_loader as data_loader
from train_value import _resolve_local_gemma_tokenizer_path
from train_value import _validate_gemma_tokenizer
from train_value import build_value_data_config


def _summarize_values(values: np.ndarray) -> str:
    values = values.astype(np.float32)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vmean = float(np.mean(values))
    vstd = float(np.std(values))
    oob = float(np.mean((values < -1.0) | (values > 0.0)))
    nans = float(np.mean(np.isnan(values)))
    infs = float(np.mean(np.isinf(values)))
    return (
        f"value[min={vmin:.4f}, max={vmax:.4f}, mean={vmean:.4f}, std={vstd:.4f}, "
        f"oob={oob:.2%}, nan={nans:.2%}, inf={infs:.2%}]"
    )


def _summarize_images(images: np.ndarray) -> str:
    images = images.astype(np.float32)
    imin = float(np.min(images))
    imax = float(np.max(images))
    imean = float(np.mean(images))
    return f"image[min={imin:.4f}, max={imax:.4f}, mean={imean:.4f}, shape={tuple(images.shape)}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 value 数据是否异常")
    parser.add_argument("--data_dir", type=str, required=True, help="LeRobot 数据集路径")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--steps", type=int, default=3, help="检查几个 batch")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker 数量")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True, help="是否打乱数据")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Gemma3 tokenizer.model 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    tokenizer_path = _resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    _validate_gemma_tokenizer(tokenizer_path)
    model_config = ValueModelConfig()
    data_config = build_value_data_config(
        args.data_dir,
        model_config,
        tokenizer_path=tokenizer_path,
    )
    loader = data_loader.create_value_data_loader(
        data_config,
        model_config,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        seed=args.seed,
        sharding=None,
        skip_norm_stats=True,
        framework="jax",
    )

    data_iter = iter(loader)
    for i in range(args.steps):
        observation, value = next(data_iter)
        obs_host = jax.device_get(observation)
        val_host = np.array(jax.device_get(value))

        img_key = next(iter(obs_host.images.keys()))
        img = np.array(obs_host.images[img_key])

        pmask = obs_host.tokenized_prompt_mask
        pmask_mean = float(np.mean(pmask)) if pmask is not None else float("nan")

        print(
            console.info(
                f"[batch {i}] {_summarize_values(val_host)}, "
                f"prompt_mask_mean={pmask_mean:.4f}, "
                f"{_summarize_images(img)}"
            )
        )


if __name__ == "__main__":
    main()
