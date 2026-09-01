# VAR Style Transfer Workspace

This workspace is for VAR-d20 style-transfer experiments.

## Structure

```text
VAR_Style_Transfer_Workspace/
  VAR/        # copied VAR source code
  content/    # content images
  style/      # style images grouped by style name
  notebooks/  # Colab notebooks
```

## Notebook Order

1. `notebooks/00_var_d20_inference_smoke_test.ipynb`
   Verifies that VAR-d20 loads and can generate samples.

2. `notebooks/01_var_d20_content_style_reconstruction.ipynb`
   Loads content/style images, tokenizes them with the VAE, and reconstructs them.

3. `notebooks/02_var_d20_vae_scale_fusion_ablation.ipynb`
   Mixes content/style VAE token scales to see which scales preserve structure and which scales carry style-like color or texture.

Do the reconstruction and scale-fusion notebooks before implementing transformer style injection.
