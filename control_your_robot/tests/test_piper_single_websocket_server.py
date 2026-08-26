import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "serve_piper_single_pi05star_websocket.py"
SPEC = importlib.util.spec_from_file_location("serve_piper_single_pi05star_websocket", SCRIPT_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SERVER)


def _complete_checkpoint(path: Path, asset_id: str) -> Path:
    (path / "params").mkdir(parents=True)
    (path / "params" / "_METADATA").touch()
    norm_dir = path / "assets" / asset_id
    norm_dir.mkdir(parents=True)
    (norm_dir / "norm_stats.json").write_text("{}", encoding="utf-8")
    return path


def test_validate_checkpoint_dir_accepts_complete_jax_checkpoint(tmp_path):
    checkpoint = _complete_checkpoint(tmp_path / "10000", "piperx_black_plug_0825_v3")

    assert SERVER._validate_checkpoint_dir(  # noqa: SLF001
        str(checkpoint), "piperx_black_plug_0825_v3"
    ) == checkpoint.resolve()


@pytest.mark.parametrize("missing", ["metadata", "norm_stats"])
def test_validate_checkpoint_dir_rejects_incomplete_checkpoint(tmp_path, missing):
    checkpoint = _complete_checkpoint(tmp_path / "10000", "piperx_black_plug_0825_v3")
    target = (
        checkpoint / "params" / "_METADATA"
        if missing == "metadata"
        else checkpoint / "assets" / "piperx_black_plug_0825_v3" / "norm_stats.json"
    )
    target.unlink()

    with pytest.raises(ValueError, match="METADATA|norm stats"):
        SERVER._validate_checkpoint_dir(str(checkpoint), "piperx_black_plug_0825_v3")  # noqa: SLF001
