import argparse
import importlib
import os
import sys
from pathlib import Path

import torch


def _ensure_numpy_core_compat() -> None:
    """Best-effort compat for loading checkpoints pickled with NumPy>=2.

    Some checkpoints (via pickle) may reference `numpy._core.*` which doesn't exist
    in NumPy 1.x; mapping it to `numpy.core.*` often allows `torch.load()` to succeed.

    This is ONLY used to load the *input* checkpoint so we can re-save a tensor-only
    weights file that is portable.
    """
    try:
        import numpy.core as numpy_core  # type: ignore
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)

    alias_map = {
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.umath": "numpy.core.umath",
    }

    for alias_name, target_name in alias_map.items():
        if alias_name in sys.modules:
            continue
        try:
            sys.modules[alias_name] = importlib.import_module(target_name)
        except Exception:
            pass


def _load_checkpoint(path: Path) -> object:
    """Load checkpoint robustly across torch versions."""
    # Prefer weights_only=False because training checkpoint contains non-tensor fields.
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        # torch<2.0 has no weights_only
        return torch.load(str(path), map_location="cpu")


def _extract_state_dict(ckpt: object, key: str) -> dict:
    """Extract a model state_dict from a training checkpoint or an already-pure file."""
    if isinstance(ckpt, dict):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
        # common alternates
        for k in ("state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]

        # Heuristic: looks like a raw state_dict already (param_name -> tensor)
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt  # type: ignore[return-value]

    raise ValueError(
        "Unable to find a model state dict in checkpoint. "
        f"Expected key '{key}' (per your training save_checkpoint), or a raw state_dict."
    )


def _tensorize_state_dict(state: dict) -> dict:
    """Ensure all values are CPU tensors (no numpy arrays etc.)."""
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            # Some older checkpoints might contain numpy arrays; convert if possible.
            try:
                import numpy as np  # type: ignore

                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v).cpu()
                    continue
            except Exception:
                pass
            raise TypeError(f"state_dict contains non-tensor value at key '{k}': {type(v)}")
    return out


def _save_pth(state: dict, out_path: Path, wrap_key: str | None) -> None:
    if wrap_key:
        torch.save({wrap_key: state}, str(out_path))
    else:
        torch.save(state, str(out_path))


def _save_safetensors(state: dict, out_path: Path) -> None:
    try:
        from safetensors.torch import save_file  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Saving safetensors requires package 'safetensors'. "
            "Install with: pip install safetensors"
        ) from e

    save_file(state, str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a training checkpoint (saved by train_lyf_with_mask.py) into a portable, "
            "tensor-only weights file to avoid numpy pickle dependency issues (e.g., numpy._core)."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input .pth checkpoint path",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="",
        help=(
            "Output path. If omitted, writes next to input as '<stem>_state_dict.pth' or .safetensors"
        ),
    )
    parser.add_argument(
        "--key",
        default="model_state_dict",
        help="Key that stores the model weights in the training checkpoint (default: model_state_dict)",
    )
    parser.add_argument(
        "--format",
        choices=["pth", "safetensors"],
        default="pth",
        help="Output format (default: pth)",
    )
    parser.add_argument(
        "--wrap-key",
        default="",
        help=(
            "If set, wraps the output as a dict under this key (e.g., model_state_dict). "
            "If empty, saves a raw state_dict (recommended)."
        ),
    )

    args = parser.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {in_path}")

    if args.out_path:
        out_path = Path(args.out_path).expanduser().resolve()
    else:
        suffix = ".safetensors" if args.format == "safetensors" else ".pth"
        out_path = in_path.with_name(in_path.stem + "_state_dict" + suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading checkpoint: {in_path}")
    try:
        ckpt = _load_checkpoint(in_path)
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") == "numpy._core":
            print("Encountered numpy._core import error; applying compat aliases and retrying...")
            _ensure_numpy_core_compat()
            ckpt = _load_checkpoint(in_path)
        else:
            raise

    print("[2/3] Extracting model_state_dict...")
    state = _extract_state_dict(ckpt, args.key)
    state = _tensorize_state_dict(state)

    print(f"  - Params: {len(state)}")
    sample_keys = list(state.keys())[:10]
    print(f"  - Sample keys: {sample_keys}")

    print(f"[3/3] Saving {args.format} -> {out_path}")
    if args.format == "pth":
        wrap_key = args.wrap_key.strip() or None
        _save_pth(state, out_path, wrap_key)
    else:
        _save_safetensors(state, out_path)

    # Quick sanity reload (pth only)
    if args.format == "pth":
        try:
            loaded = torch.load(str(out_path), map_location="cpu")
            if isinstance(loaded, dict) and args.wrap_key and args.wrap_key in loaded:
                loaded = loaded[args.wrap_key]
            ok = isinstance(loaded, dict) and len(loaded) == len(state)
            print(f"Sanity reload: {'OK' if ok else 'CHECK'}")
        except Exception as e:
            print(f"Sanity reload failed (non-fatal): {e}")

    print("Done.")
    print("Tip: Copy the output file to Jetson and point REID_MODEL_PATH to it.")


if __name__ == "__main__":
    main()
