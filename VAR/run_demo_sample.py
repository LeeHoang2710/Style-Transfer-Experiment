import os.path as osp
from pathlib import Path

import torch
import torchvision
from PIL import Image

from models import build_vae_var


MODEL_DEPTH = 16
SEED = 0
CLASS_LABELS = (980, 437, 22, 562)
CFG = 4.0
TOP_K = 900
TOP_P = 0.95
MORE_SMOOTH = False
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
OUT = ROOT / "outputs" / "var_d16_samples"


def main() -> None:
    assert MODEL_DEPTH in {16, 20, 24, 30}
    assert torch.cuda.is_available(), "CUDA is required for this smoke test."

    device = "cuda"
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae_ckpt = WEIGHTS / "vae_ch160v4096z32.pth"
    var_ckpt = WEIGHTS / f"var_d{MODEL_DEPTH}.pth"
    OUT.mkdir(parents=True, exist_ok=True)

    if not osp.exists(vae_ckpt):
        raise FileNotFoundError(f"Missing {vae_ckpt}")
    if not osp.exists(var_ckpt):
        raise FileNotFoundError(f"Missing {var_ckpt}")

    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=patch_nums,
        num_classes=1000,
        depth=MODEL_DEPTH,
        shared_aln=False,
    )

    vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
    vae.eval()
    var.eval()

    torch.manual_seed(SEED)
    label_b = torch.tensor(CLASS_LABELS, device=device)

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
            recon = var.autoregressive_infer_cfg(
                B=len(CLASS_LABELS),
                label_B=label_b,
                cfg=CFG,
                top_k=TOP_K,
                top_p=TOP_P,
                g_seed=SEED,
                more_smooth=MORE_SMOOTH,
            )

    grid = torchvision.utils.make_grid(recon, nrow=len(CLASS_LABELS), padding=0, pad_value=1.0)
    grid = grid.permute(1, 2, 0).mul_(255).cpu().numpy()
    image = Image.fromarray(grid.astype("uint8"))
    out_path = osp.abspath(OUT / "var_d16_sample_grid.png")
    image.save(out_path)
    print(out_path)
    print(f"shape={image.size}, cuda_max_memory_mb={torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")


if __name__ == "__main__":
    main()
