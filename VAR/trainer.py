import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    """
    Sample input for this class trainer:
    device = "cuda:0"
    patch_nums=(1,2,3,4,5,6,8,10,13,16),
    resos=(16,32,48,64,80,96,128,160,208,256),
    vae_local=VQVAE pretrained on ImageNet-1k, load from checkpoint (as VAR training does not train VQVAE)

    what is var_wo_ddp and var? var_wo_ddp is the VAR model without DDP, var is the VAR model with DDP. var_opt is the optimizer for var_wo_ddp.
    what is DDP? DDP is the DistributedDataParallel module from PyTorch, which allows for distributed training across multiple GPUs and nodes. It wraps the model and handles the communication between different processes during training.
    we need var_wo_ddp because we need to access the model without DDP for certain operations, such as saving and loading checkpoints, and for evaluation. DDP is used during training to synchronize gradients across multiple GPUs and nodes.
    """
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth # what is label_smooth? label_smooth is a technique used to improve the generalization of the model by softening the target labels. Instead of using hard labels (0 or 1), we use smoothed labels (e.g., 0.9 for the correct class and 0.1 for the incorrect classes). This helps prevent the model from becoming overconfident and improves its ability to generalize to unseen data.
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none') # apply label smoothing to the training loss, but not to the validation loss, as we want to evaluate the model's performance on the original hard labels.
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums) # for default L (num of tokens across all scales) = 680 and last_l (num of tokens in the final scale) = 16*16 = 256
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L # what is loss_weight? loss_weight is a tensor that contains the weights for each token in the loss calculation. It is used to give different importance to different tokens during training. In this case, we are giving equal weight to all tokens by setting the weights to 1/L, where L is the total number of tokens across all scales. This means that each token contributes equally to the overall loss.
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn)) # [(0, 1), (1, 5), (5, 14), (14, 30), (30, 55), (55, 91), (91, 155), (155, 255), (255, 464), (464, 680)]
            cur += pn*pn
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0 # average loss and accuracy across all scales (L_mean, acc_mean) and for the last scale only (L_tail, acc_tail)
        stt = time.time()
        training = self.var_wo_ddp.training # store the current training state of the model (True if in training mode, False if in evaluation mode)
        self.var_wo_ddp.eval() # eval mode disables dropout and batch normalization layers, which are only used during training. This ensures that the model is evaluated in a consistent manner.
        for inp_B3HW, label_B in ld_val: # load the input images (B, 3, H, W) and the corresponding labels (B) from the validation DataLoader. 
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True) # B, 3, H, W
            label_B = label_B.to(dist.get_device(), non_blocking=True) # B
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW) # [B, 1, 1], [B, 2, 2], [B, 3, 3], [B, 4, 4], [B, 5, 5], [B, 6, 6], [B, 8, 8], [B, 10, 10], [B, 13, 13], [B, 16, 16]
            gt_BL = torch.cat(gt_idx_Bl, dim=1) # B, L (B, 680)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # B, 679, Cvae (B, 679, 32) 
            # VAR does not need previous visual tokens to predict the first scale. It predicts the first scale from the class/start embedding.
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen],
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW) # [B, 1], [B, 4], ..., [B, 256]
        gt_BL = torch.cat(gt_idx_Bl, dim=1) # B, L (B, 680)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        
        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            loss = loss.mul(self.loss_weight).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        # log to tensorboard
        if g_it == 0 or (g_it + 1) % 500 == 0:
            prob_per_class_is_chosen = pred_BL.view(-1).bincount(minlength=V).float()
            dist.allreduce(prob_per_class_is_chosen)
            prob_per_class_is_chosen /= prob_per_class_is_chosen.sum()
            cluster_usage = (prob_per_class_is_chosen > 0.001 / V).float().mean().item() * 100
            if dist.is_master():
                if g_it == 0:
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-10000)
                    tb_lg.update(head='AR_iter_loss', z_voc_usage=cluster_usage, step=-1000)
                kw = dict(z_voc_usage=cluster_usage)
                for si, (bg, ed) in enumerate(self.begin_ends):
                    pred, tar = logits_BLV.data[:, bg:ed].reshape(-1, V), gt_BL[:, bg:ed].reshape(-1)
                    acc = (pred.argmax(dim=-1) == tar).float().mean().item() * 100
                    ce = self.val_loss(pred, tar).item()
                    kw[f'acc_{self.resos[si]}'] = acc
                    kw[f'L_{self.resos[si]}'] = ce
                tb_lg.update(head='AR_iter_loss', **kw, step=g_it)
        
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
