from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

import dist


# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]
""" 
    In normal VQ-VAE, encoder output f -> find nearest codebook vector at every 16x16 location -> get token grid 16x16 -> reconstruct f_hat from those tokens.
    VAR does not only create 16x16 token grid but also create 8x8, 4x4, 2x2, 1x1 token grids. So the quantizer does not quantize the same feature map independently at every scale. Instead, it works like this:
    - f_rest = original encoder output f and f_hat = 0 (line 73, 74)
    At scale 1x1: 
    - find nearest codebook vector at 1x1 location -> get token grid 1x1 -> f_hat += f_hat1 and f_rest = f - f_hat1 (line 100, 101)
    - f_rest contains what the 1x1 scale failed to explain
    At scale 2x2:
    - find nearest codebook vector that approximates f_rest at 2x2 locations -> get token grid 2x2 -> f_hat += f_hat2 and f_rest = f - f_hat1 - f_hat2
    - f_rest contains what the 1x1 and 2x2 scales failed to explain
"""


class VectorQuantizer2(nn.Module):
    """
    continuous encoder feature map -> discrete codebook index map at multiple scales -> reconstruct latent feature map
    Encoder output example: B, Cvae, H, W = B, 32, 32, 32
    """
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size # vocab_size is the number of discrete codes in the codebook, 4096
        self.Cvae: int = Cvae # channel size of each code vector, 32
        self.using_znorm: bool = using_znorm # whether to use z-normalization for the feature map before quantization, True
        self.v_patch_nums: Tuple[int] = v_patch_nums # scale size of the feature map at each scale, e.g., (1, 2, 4, 8, 16)
        
        self.quant_resi_ratio = quant_resi # Different scales of the feature map might need different levels of adaptation after quantization. This parameter controls how much of the original codebook vector and how much of the adapted vector to use. 0.5 means equal weighting.
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales -> each scale has its own Phi
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales -> all scales share the same Phi
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales -> scales share Phi in a round-robin manner
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae) # randomly initialize embedding layer that maps discrete codebook indices to continuous code vectors -> trainable
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BChw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape # encoder output feature map shape, e.g., B, 32, 32, 32
        f_no_grad = f_BChw.detach() # a copy of the original feature map. As nearest codebook vector search is non-differentiable, and gradient should not flow through this f_no_grad back to the encoder
        
        f_rest = f_no_grad.clone() # remaining error after each scale and non-differentiable
        f_hat = torch.zeros_like(f_rest) # accumulated reconstruction from each scale and no gradient history
        
        with torch.cuda.amp.autocast(enabled=False): # only float32
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BChw.device) # how often each codebook vector is used across all scales
            SN = len(self.v_patch_nums) # number of scales, e.g., 5 for 1x1, 2x2, 4x4, 8x8, 16x16
            for si, pn in enumerate(self.v_patch_nums): # from small to large scales: si = 0, 1, 2, 3, 4; pn = 1x1, 2x2, 4x4, 8x8, 16x16
                rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # downsample f_rest to the current scale size, then flatten to (B*pn*pn, C) for nearest codebook vector search. If it's the last scale, use the original f_rest without downsampling.

                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # normalize both feature and embedding codebook vectors -> pick the embedding with the highest cosine similarity (dot product) as the nearest codebook vector
                else:
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False) # compute the squared L2 distance between each feature vector and each embedding codebook vector. d_no_grad = ||x||^2 + ||y||^2 
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size) -> d_no_grad -= 2 * x @ y^T
                    idx_N = torch.argmin(d_no_grad, dim=1) #  d_no_grad = ||x||^2 + ||y||^2 - 2 * x @ y^T = ||x - y||^2
                
                hit_V = idx_N.bincount(minlength=self.vocab_size).float() # how many times each codebook vector is used at this scale
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn) # convert flatten ID back to 2D grid of size pn x pn. For ex: B*pn*pn -> B, pn, pn
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous() # B, pn, pn -> B, C, pn, pn -> upsample to B, C, H, W. This is the reconstructed feature map from the nearest codebook vectors at this scale.
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # apply the Phi to adapt the codebook vector to its surroundings -> differentiable
                f_hat = f_hat + h_BChw # accumlated reconstruction from each scale -> differentiable
                f_rest -= h_BChw # remaining error after each scale
                
                if self.training and dist.initialized():
                    handler.wait()
                    # update the exponential moving average of codebook vector usage across all scales. This is used to monitor how many codebook vectors are being used and to encourage diversity in the codebook usage.
                    # early training: EMA = 0.9old + 0.1new; later training: EMA = 0.99old + 0.01new
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad) 
                """
                The loss has 2 terms: 
                F.mse_loss(f_hat.data, f_BChw) -> gradient go to: f_BChw to encoder
                F.mse_loss(f_hat, f_no_grad) -> gradient go to: f_hat to embedding codebook and Phi
                The commitment loss: tell the encoder stay close to the codebook vector you selected, so that the encoder output does not drift away from the codebook vectors.
                The codebook loss: tell the codebook vector to stay close to the encoder output, so that the codebook vectors do not drift away from the encoder output.
                """
            
            mean_vq_loss *= 1. / SN # normalize the loss by the number of scales, so that the loss is not biased towards larger scales.
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw) # straight-through estimator
            """
            f_no_grad = f_BChw.detach() and f_hat.data are both non-differentiable
            However we want the gradient to flow from decoder -> f_hat -> f_BChw -> the encoder, so we add f_BChw (differentiable) to restores gradient flow to the encoder. 
            gradient go to: f_hat to encoder
            """
        
        margin = tdist.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: usages = None
        # f_hat is the reconstructed latent feature map
        # usages is the percentage of codebook vectors used at each scale
        # mean_vq_loss is the average quantization loss across all scales
        return f_hat, usages, mean_vq_loss
    # ===================== `forward` is only used in VAE training =====================

    """
    1. f_to_idxBl_or_fhat: continuous latent feature f (before quantization) -> list of discrete token IDs or list of reconstructed latent feature f_hat at each scale.
    2. embed_to_fhat: multi-scale embedding maps (after quantization) -> list of reconstructed latent feature f_hat at each scale or final reconstructed latent feature f_hat.
    3. idxBl_to_var_input: ground truth discrete token IDs at each scale -> multi-scale embedding maps (after quantization) -> VAR training input features B, L, C (L = sum of all scales' token counts)
    """
    
    def embed_to_fhat(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        multi-scale embedding maps -> multi-scale reconstructed latent feature maps
        Input is a list of embedding maps at different scales, each of shape B, C, pn, pn. [B, C, 1, 1], [B, C, 2, 2], [B, C, 4, 4], [B, C, 8, 8], [B, C, 16, 16]
        Output is a list of reconstructed latent feature maps at different scales, each of shape B, C, H, W. [B, C, 16, 16], [B, C, 16, 16], [B, C, 16, 16], [B, C, 16, 16], [B, C, 16, 16]

        f_hat = f_hat = Phi(1x1 upsampled) + Phi(2x2 upsampled) + Phi(4x4 upsampled) + Phi(8x8 upsampled) + Phi(16x16)
        """
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1] # final latent size. If the last scale is 16x16 -> H = W = 16
        SN = len(self.v_patch_nums)
        if all_to_max_scale: # make all scales to the max scale, e.g., 16x16. This is the default behavior in VQ-VAE training and inference.
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, H, W, dtype=torch.float32) # shape B, C, H, W. This is the accumulated reconstructed latent feature map from all scales.
            for si, pn in enumerate(self.v_patch_nums): # from small to large scales: si = 0, 1, 2, 3, 4; pn = 1x1, 2x2, 4x4, 8x8, 16x16
                h_BChw = ms_h_BChw[si] # shape B, C, pn, pn. This is the embedding map at the current scale.
                if si < len(self.v_patch_nums) - 1: # if not the last scale, upsample to the max scale, e.g., 16x16.
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # apply the Phi to adapt the codebook vector to its surroundings
                f_hat.add_(h_BChw) # accumlated reconstruction from each scale
                if last_one: ls_f_hat_BChw = f_hat # if last_one is True, return only the final reconstructed latent feature map. This is used in VAR inference, where we only need the final output.
                else: ls_f_hat_BChw.append(f_hat.clone()) # if last_one is False, return the reconstructed latent feature map at each scale. This is used in VAR training, where we need the output at each scale for loss computation.
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode='bicubic')
                h_BChw = self.quant_resi[si/(SN-1)](ms_h_BChw[si])
                f_hat.add_(h_BChw)
                if last_one: ls_f_hat_BChw = f_hat
                else: ls_f_hat_BChw.append(f_hat)
        
        return ls_f_hat_BChw
    
    def f_to_idxBl_or_fhat(self, f_BChw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[Union[torch.Tensor, torch.LongTensor]]:  # z_BChw is the feature from inp_img_no_grad
        """
        if to_fhat is True:
            continuous latent feature f -> multi-scale reconstructed latent feature f_hat
        else:
            continuous latent feature f -> multi-scale discrete token IDs
        """
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        # what is the different between detach and clone? detach() returns a new tensor that shares the same data but does not require gradients, while clone() returns a new tensor that is a copy of the original tensor and requires gradients. In this case, f_no_grad is used to compute the loss without affecting the gradients of f_BChw, while f_rest is used to accumulate the remaining error after each scale and requires gradients for backpropagation.
        # why we not write f_rest = f_BChw.clone()? Because we want to compute the loss between f_hat and f_BChw, and f_rest is used to accumulate the remaining error after each scale. If we use f_BChw.clone(), the gradients will flow back to f_BChw and affect the encoder output, which is not what we want. By using f_no_grad.clone(), we ensure that the gradients do not flow back to f_BChw and only affect the codebook and Phi parameters.
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]    # from small to large [(1, 1), ..., (16, 16)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(ph, pw), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C) # B*ph*pw, C
            # so we have to find the nearest embedding for each of the B*ph*pw feature vectors in z_NC.
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1) # B*ph*pw, C @ C, vocab_size -> B*ph*pw, vocab_size -> argmax -> B*ph*pw
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_N = torch.argmin(d_no_grad, dim=1) # L2 distance: find the nearest embedding vector for each feature vector in z_NC
            
            idx_Bhw = idx_N.view(B, ph, pw) # convert flatten token IDs back to 2D grid of size ph x pw. For ex: B*ph*pw -> B, ph, pw
            h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous() # B, ph, pw -> B, C, ph, pw -> upsample to B, C, H, W. This is the reconstructed feature map from the nearest codebook vectors at this scale.
            h_BChw = self.quant_resi[si/(SN-1)](h_BChw) # apply the Phi to adapt the codebook vector to its surroundings
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph*pw))
        
        return f_hat_or_idx_Bl
    
    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        """
        gt_ms_idx_Bl = vae.img_to_idxBl(image) -> 5 scales of discrete token IDs, each of shape B, pn*pn. For example: [B, 1], [B, 4], [B, 16], [B, 64], [B, 256]
        grouth truth multi-scale token IDs -> continuous VQVAE latent features used for teacher-forcing input for VAR training
        """
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        
        f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN-1):
            h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic') # [B, 1] -> [B, 1, 32] -> [B, 32, 1] -> [B, 32, 1, 1] -> [B, 32, 16, 16]
            f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
            pn_next = self.v_patch_nums[si+1]
            next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        """
        during inference, after VAR predicts current scale, update f_hat and f_rest. Prepare input = f_hat for the next scale 
        """
        HW = self.v_patch_nums[-1]
        if si != SN-1: # current scale is not the last scale
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic')) # upsample current predicted scale to full latent size, then apply Phi to adapt the codebook vector to its surroundings
            f_hat.add_(h) # add to accumulated f_hat
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area') # downsample to next scale's size, this will be the input for the next scale
        else: # current scale is the last scale
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat # no next scale, the final f_hat is ready for VQVAE decoder


class Phi(nn.Conv2d):
    """
    Phi is a small 3x3 convolutional layer applied after codebook lookup.
    Because codebook lookup is rigid and a codebook vector is the same for all locations.
    But images need local adaptation, as the same visual token might need to behave slightly differently depending on its neighbors. 
    So Phi lets the model locally adapt the codebook vector to its surroundings.
    """
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        # output = (1 - resi_ratio) * original_codebook_vector + resi_ratio * Phi(original_codebook_vector)
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
