import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

from models import build_vae_var


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights"
OUT = ROOT / "outputs" / "var_d20_samples"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DEPTH = 20
PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 12 samples, intentionally more than the notebook's small 4-image examples.
CLASS_LABELS = [
    980,  # volcano
    437,  # beacon/lighthouse-ish class in ImageNet ids
    22,
    562,
    281,
    207,
    285,
    151,
    323,
    409,
    852,
    954,
]


def report_memory(stage: str) -> None:
    if DEVICE != "cuda":
        print(f"{stage}: cpu")
        return
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{stage}: allocated={used:.2f} GB reserved={reserved:.2f} GB peak={peak:.2f} GB")


def main() -> None:
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"device={DEVICE}")
    if DEVICE == "cuda":
        print(torch.cuda.get_device_name(0))

    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=DEVICE,
        patch_nums=PATCH_NUMS,
        num_classes=1000,
        depth=MODEL_DEPTH,
        shared_aln=False,
    )

    vae.load_state_dict(torch.load(WEIGHTS / "vae_ch160v4096z32.pth", map_location="cpu"), strict=True)
    var.load_state_dict(torch.load(WEIGHTS / "var_d20.pth", map_location="cpu"), strict=True)
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    report_memory("after_load")

    images = []
    start = time.time()
    with torch.inference_mode():
        for i, label in enumerate(CLASS_LABELS, start=1):
            label_tensor = torch.tensor([label], device=DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda", dtype=torch.float16, cache_enabled=True):
                sample = var.autoregressive_infer_cfg(
                    B=1,
                    label_B=label_tensor,
                    cfg=4,
                    top_k=900,
                    top_p=0.95,
                    g_seed=seed + i,
                    more_smooth=False,
                )
            image = sample[0].detach().cpu()
            torchvision.utils.save_image(image, OUT / f"{i:02d}_class_{label}.png")
            images.append(image)
            report_memory(f"sample_{i:02d}")

    grid = torchvision.utils.make_grid(torch.stack(images), nrow=4, padding=2, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp_(0, 255).byte().numpy()
    Image.fromarray(grid_np).save(OUT / "grid_12_samples.png")
    print(f"saved={OUT}")
    print(f"elapsed_sec={time.time() - start:.1f}")
    report_memory("done")


if __name__ == "__main__":
    main()
