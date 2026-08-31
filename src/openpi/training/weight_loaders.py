import dataclasses
import logging
import os
import re
import sys
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import flax
import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.shared import console

logger = logging.getLogger(__name__)

_BV_PATH = os.path.join(os.path.dirname(__file__), "../../../big_vision")
if _BV_PATH not in sys.path:
    sys.path.insert(0, _BV_PATH)
try:
    from big_vision import utils as bv_utils  # type: ignore
    from big_vision.models import vit as bv_vit  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bv_utils = None
    bv_vit = None

try:
    from gemma.gm import ckpts as gm_ckpts  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gm_ckpts = None

try:
    from gemma.gm.ckpts import _checkpoint as gm_ckpt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    gm_ckpt = None


def _flatten_keys(tree: at.Params) -> set[str]:
    try:
        flat = flax.traverse_util.flatten_dict(tree, sep="/")
    except Exception:
        return set()
    return set(flat.keys())


def _summarize_param_match(name: str, loaded_params: at.Params, ref_params: at.Params) -> None:
    """Log a short compatibility summary between loaded params and reference params."""
    try:
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(ref_params, sep="/")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(console.warn(f"{name} params: failed to flatten for comparison ({exc})"))
        return

    loaded_keys = set(flat_loaded.keys())
    ref_keys = set(flat_ref.keys())
    overlap = loaded_keys & ref_keys
    missing = ref_keys - loaded_keys
    extra = loaded_keys - ref_keys

    shape_mismatch = []
    for k in overlap:
        lv = flat_loaded.get(k)
        rv = flat_ref.get(k)
        if hasattr(lv, "shape") and hasattr(rv, "shape") and lv.shape != rv.shape:
            shape_mismatch.append(k)

    logger.info(
        console.info(
            f"{name} params: matched {len(overlap)}/{len(ref_keys)} "
            f"(missing {len(missing)}, extra {len(extra)}, shape_mismatch {len(shape_mismatch)})"
        )
    )
    if missing:
        sample = list(sorted(missing))[:10]
        logger.warning(console.warn(f"{name} missing keys (sample): {sample}"))
    if shape_mismatch:
        sample = list(sorted(shape_mismatch))[:10]
        logger.warning(console.warn(f"{name} shape mismatch keys (sample): {sample}"))


def _validate_param_coverage(name: str, loaded_params: at.Params, ref_params: at.Params) -> None:
    """Require every reference parameter to be present with the expected shape."""
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(ref_params, sep="/")
    missing = sorted(set(flat_ref) - set(flat_loaded))
    shape_mismatch = []
    for key in set(flat_ref) & set(flat_loaded):
        loaded_value = flat_loaded[key]
        ref_value = flat_ref[key]
        if hasattr(ref_value, "shape") and (
            not hasattr(loaded_value, "shape")
            or not hasattr(loaded_value, "dtype")
            or loaded_value.shape != ref_value.shape
        ):
            shape_mismatch.append(key)
    shape_mismatch.sort()
    if missing or shape_mismatch:
        matched = len(flat_ref) - len(missing) - len(shape_mismatch)
        raise ValueError(
            f"{name} checkpoint is incompatible: matched {matched}/{len(flat_ref)}, "
            f"missing={len(missing)}, shape_mismatch={len(shape_mismatch)}; "
            f"missing sample={missing[:10]}, shape mismatch sample={shape_mismatch[:10]}"
        )


def _select_subtree_by_overlap(loaded_params: at.Params, ref_params: at.Params, name: str) -> at.Params:
    """Select a nested subtree with best key overlap against ref params."""
    if not isinstance(loaded_params, Mapping):
        return loaded_params
    ref_keys = _flatten_keys(ref_params)
    if not ref_keys:
        return loaded_params

    def _overlap(p: at.Params) -> int:
        return len(_flatten_keys(p) & ref_keys)

    current = loaded_params
    current_score = _overlap(current)
    improved = True
    while improved and isinstance(current, Mapping):
        improved = False
        best_key = None
        best_params = None
        best_score = current_score
        for k, v in current.items():
            if isinstance(v, Mapping):
                score = _overlap(v)
                if score > best_score:
                    best_score = score
                    best_key = k
                    best_params = v
        if best_params is not None:
            logger.info(console.info(f"{name} params: selecting '{best_key}' subtree (overlap={best_score})"))
            current = best_params
            current_score = best_score
            improved = True
    return current


def _maybe_resample_siglip_posemb(siglip_params: at.Params, ref_params: at.Params) -> at.Params:
    """Resample SigLIP position embedding to match reference shape when needed."""
    if bv_vit is None:
        return siglip_params
    if not isinstance(siglip_params, Mapping) or not isinstance(ref_params, Mapping):
        return siglip_params

    def _get_posemb(p: Mapping):
        if "pos_embedding" in p:
            return ("pos_embedding", p["pos_embedding"])
        if "Transformer" in p and isinstance(p["Transformer"], Mapping) and "pos_embedding" in p["Transformer"]:
            return ("Transformer/pos_embedding", p["Transformer"]["pos_embedding"])
        return (None, None)

    src_key, src = _get_posemb(siglip_params)
    dst_key, dst = _get_posemb(ref_params)
    if src_key is None or dst_key is None:
        return siglip_params
    if not hasattr(src, "shape") or not hasattr(dst, "shape") or src.shape == dst.shape:
        return siglip_params

    try:
        resized = bv_vit.resample_posemb(src, dst)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(console.warn(f"SigLIP pos_embedding resize failed: {exc}"))
        return siglip_params

    if src_key == "pos_embedding":
        siglip_params = dict(siglip_params)
        siglip_params["pos_embedding"] = resized
        return siglip_params

    # Transformer/pos_embedding
    siglip_params = dict(siglip_params)
    transformer = siglip_params.get("Transformer")
    if isinstance(transformer, Mapping):
        transformer = dict(transformer)
        transformer["pos_embedding"] = resized
        siglip_params["Transformer"] = transformer
    return siglip_params


def _maybe_convert_gemma_ckpt_tree(tree: at.Params) -> at.Params:
    """Convert Gemma checkpoint tree to nested format if needed."""
    converted = tree
    if gm_ckpt is not None:
        try:
            ckpt_tree = gm_ckpt._CheckpointTree(tree=tree)
            converted = ckpt_tree.nested_tree
        except Exception as exc:  # pragma: no cover - version-dependent fallback
            logger.warning(console.warn(f"Gemma checkpoint convert failed: {exc}; using prefix fallback"))

    if not isinstance(converted, Mapping):
        return converted

    flat = flax.traverse_util.flatten_dict(converted, sep="/")
    prefix = "transformer/"
    if flat and all(key.startswith(prefix) for key in flat):
        renamed = {}
        for key, value in flat.items():
            key = key[len(prefix) :]
            if key.endswith(("/mlp/gating_einsum/w", "/mlp/linear/w")):
                key = key.removesuffix("/w")
            if key in renamed:
                raise ValueError(f"Gemma checkpoint key collision after normalization: {key}")
            renamed[key] = value
        converted = flax.traverse_util.unflatten_dict(
            renamed,
            sep="/",
        )
        logger.info(console.info(f"Gemma checkpoint: stripped '{prefix}' from {len(flat)} parameters"))
    return converted


def _select_siglip_image_params(siglip_params: at.Params, ref_params: at.Params) -> at.Params:
    """Select the SigLIP image tower params from a checkpoint tree."""
    if not isinstance(siglip_params, Mapping):
        return siglip_params

    img_ref_keys = {k[len("img/") :] for k in _flatten_keys(ref_params) if k.startswith("img/")}

    candidates: list[tuple[str, at.Params]] = [("root", siglip_params)]
    for key in ("img", "image", "vision", "visual", "image_encoder", "vision_encoder", "params"):
        if key in siglip_params and isinstance(siglip_params[key], Mapping):
            candidates.append((key, siglip_params[key]))
    if len(siglip_params) == 1:
        only_key = next(iter(siglip_params))
        only_val = siglip_params[only_key]
        if isinstance(only_val, Mapping):
            candidates.append((only_key, only_val))

    if not img_ref_keys:
        for name, cand in candidates:
            if name == "img":
                logger.info(console.info("SigLIP image params: fallback to 'img' subtree"))
                return cand
        return siglip_params

    best_overlap = -1
    best_name = "root"
    best_params = siglip_params
    for name, cand in candidates:
        overlap = len(_flatten_keys(cand) & img_ref_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_params = cand

    if best_name != "root":
        logger.info(console.info(f"SigLIP image params: using '{best_name}' subtree (overlap={best_overlap})"))
    return best_params


def _stack_params(blocks: list[at.Params]) -> at.Params:
    if not blocks:
        return {}
    first = blocks[0]
    if isinstance(first, Mapping):
        common_keys = set(first.keys())
        for b in blocks[1:]:
            if isinstance(b, Mapping):
                common_keys &= set(b.keys())
            else:
                return first
        if common_keys != set(first.keys()):
            logger.warning(console.warn("SigLIP encoderblock keys mismatch across layers; dropping non-common keys"))
        return {k: _stack_params([b[k] for b in blocks]) for k in sorted(common_keys)}
    return np.stack(blocks, axis=0)


def _maybe_convert_siglip_to_scan(siglip_params: at.Params) -> at.Params:
    """Convert unscanned SigLIP encoderblock_* params to scanned encoderblock."""
    if not isinstance(siglip_params, Mapping):
        return siglip_params
    transformer = siglip_params.get("Transformer")
    if not isinstance(transformer, Mapping):
        return siglip_params
    if "encoderblock" in transformer:
        return siglip_params

    block_keys = [k for k in transformer.keys() if re.fullmatch(r"encoderblock_\\d+", k)]
    if not block_keys:
        return siglip_params

    block_keys.sort(key=lambda k: int(k.split("_")[1]))
    blocks = [transformer[k] for k in block_keys]
    stacked = _stack_params(blocks)

    new_transformer = dict(transformer)
    for k in block_keys:
        new_transformer.pop(k, None)
    new_transformer["encoderblock"] = stacked

    new_params = dict(siglip_params)
    new_params["Transformer"] = new_transformer
    logger.info(console.info(f"SigLIP params: converted {len(block_keys)} encoder blocks to scanned format"))
    return new_params


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*", log_prefix="Checkpoint")




@dataclasses.dataclass(frozen=True)
class ValueModelWeightLoader(WeightLoader):
    """加载 SigLIP 和 Gemma 3 270M 预训练权重用于 ValueModel。

    - SigLIP: 
    - Gemma 3 270M: 
    - ValueHead: 随机初始化
    """

    gemma_variant: str = "gemma3-270m"
    siglip_path: str | None = None
    gemma_checkpoint_dir: str | None = None

    def load(self, params: at.Params) -> at.Params:
        logger.info(console.info("加载 SigLIP 权重 (from local checkpoint)..."))
        siglip_path = self.siglip_path or os.getenv("OPENPI_VALUE_SIGLIP_PATH")
        if not siglip_path:
            raise ValueError(
                "SigLIP checkpoint path is required. Pass --siglip_path or set "
                "OPENPI_VALUE_SIGLIP_PATH."
            )
        siglip_path = os.path.abspath(os.path.expanduser(siglip_path))
        if not os.path.isfile(siglip_path):
            raise FileNotFoundError(f"SigLIP checkpoint not found: {siglip_path}")
        siglip_params = None
        if bv_utils is not None:
            try:
                siglip_params = bv_utils.load_params(f"{siglip_path}:img")
                logger.info(console.info("SigLIP params loaded via big_vision.utils.load_params(:img)"))
            except Exception as e:
                logger.warning(console.warn(f"SigLIP load via big_vision.utils failed: {e}; fallback to np.load"))
                siglip_params = None
        if siglip_params is None:
            with open(siglip_path, "rb") as f:
                siglip_flat = dict(np.load(f, allow_pickle=False))
            siglip_tree = flax.traverse_util.unflatten_dict(siglip_flat, sep="/")
            siglip_params = siglip_tree["params"] if "params" in siglip_tree else siglip_tree
        if isinstance(siglip_params, flax.core.FrozenDict):
            siglip_params = flax.core.unfreeze(siglip_params)
        siglip_params = _select_siglip_image_params(siglip_params, params)
        siglip_params = _maybe_convert_siglip_to_scan(siglip_params)
        if isinstance(params, Mapping) and "img" in params:
            siglip_params = _maybe_resample_siglip_posemb(siglip_params, params["img"])
        if isinstance(params, Mapping) and "img" in params:
            _summarize_param_match("SigLIP", siglip_params, params["img"])
            _validate_param_coverage("SigLIP", siglip_params, params["img"])

        logger.info(console.info("加载 Gemma 3 270M 权重 (from local Orbax checkpoint)..."))
        gemma_checkpoint_dir = self.gemma_checkpoint_dir or os.getenv("OPENPI_VALUE_GEMMA_CKPT")
        if not gemma_checkpoint_dir:
            raise ValueError(
                "Gemma checkpoint path is required. Pass --gemma_checkpoint_dir or set "
                "OPENPI_VALUE_GEMMA_CKPT."
            )
        gemma_checkpoint_dir = os.path.abspath(os.path.expanduser(gemma_checkpoint_dir))
        if not os.path.isdir(gemma_checkpoint_dir):
            raise FileNotFoundError(f"Gemma checkpoint directory not found: {gemma_checkpoint_dir}")

        # 使用 Gemma 官方 ckpts loader（优先），必要时回退 Orbax
        try:
            gemma_params = None
            if gm_ckpts is not None and isinstance(params, Mapping) and "llm" in params:
                try:
                    gemma_params = gm_ckpts.load_params(
                        gemma_checkpoint_dir,
                        params=params["llm"],
                        donate=False,
                    )
                    logger.info(console.info("Gemma params loaded via gemma.gm.ckpts.load_params"))
                except Exception as e:
                    logger.warning(console.warn(f"Gemma ckpts loader failed: {e}; fallback to Orbax"))
                    gemma_params = None

            if gemma_params is None:
                from orbax.checkpoint import PyTreeCheckpointer

                checkpointer = PyTreeCheckpointer()
                gemma_params = checkpointer.restore(gemma_checkpoint_dir)
                if isinstance(gemma_params, flax.core.FrozenDict):
                    gemma_params = flax.core.unfreeze(gemma_params)
                gemma_params = _maybe_convert_gemma_ckpt_tree(gemma_params)
                # 兼容不同checkpoint结构：先尝试挑选与参考参数匹配度最高的子树
                if isinstance(params, Mapping) and "llm" in params:
                    gemma_params = _select_subtree_by_overlap(gemma_params, params["llm"], "Gemma")
            logger.info(console.ok("成功加载 Gemma checkpoint"))
        except Exception as e:
            logger.error(console.error(f"Gemma 加载失败: {e}"))
            raise
        if isinstance(params, Mapping) and "llm" in params:
            _summarize_param_match("Gemma", gemma_params, params["llm"])
            _validate_param_coverage("Gemma", gemma_params, params["llm"])

        loaded_params = {
            "img": siglip_params,
            "llm": gemma_params,
        }

        # 匹配 value_head, img_projection, cross_attention, cross_attn_norm
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*(value_head|img_projection|cross_att).*",
            log_prefix="ValueModel",
        )


def _merge_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    log_prefix: str | None = None,
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    loaded_keys = set()
    shape_mismatch = []
    dtype_cast = 0
    for k, v in flat_loaded.items():
        if k in flat_ref:
            loaded_keys.add(k)
            ref_v = flat_ref[k]
            if hasattr(v, "shape") and hasattr(ref_v, "shape") and v.shape != ref_v.shape:
                shape_mismatch.append((k, getattr(v, "shape", None), getattr(ref_v, "shape", None)))
                continue
            if v.dtype != ref_v.dtype:
                dtype_cast += 1
                result[k] = v.astype(ref_v.dtype)
            else:
                result[k] = v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    allowed_missing = {k for k in flat_ref if pattern.fullmatch(k)}
    for k in allowed_missing:
        if k not in result:
            result[k] = flat_ref[k]

    # Logging: coverage, missing, extras, shape mismatches.
    if log_prefix is None:
        log_prefix = "Weights"
    missing = set(flat_ref.keys()) - set(result.keys())
    unexpected = loaded_keys - set(flat_ref.keys())
    if missing:
        # Only show a small sample to avoid log spam.
        sample = sorted(missing)[:10]
        logger.warning(
            console.warn(
                f"{log_prefix} missing keys: {len(missing)} (sample: {sample})"
            )
        )
    if unexpected:
        sample = sorted(unexpected)[:10]
        logger.warning(
            console.warn(
                f"{log_prefix} unexpected loaded keys: {len(unexpected)} (sample: {sample})"
            )
        )
    if shape_mismatch:
        sample = shape_mismatch[:5]
        logger.warning(
            console.warn(
                f"{log_prefix} shape mismatch: {len(shape_mismatch)} (sample: {sample})"
            )
        )
    if dtype_cast:
        logger.info(console.info(f"{log_prefix} dtype-cast params: {dtype_cast}"))

    return flax.traverse_util.unflatten_dict(result, sep="/")
