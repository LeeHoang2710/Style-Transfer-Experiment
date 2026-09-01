import torch
import torch.nn as nn
import torch.nn.functional as F


# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder',]


"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""
# swish
def nonlinearity(x):
    return x * torch.sigmoid(x)

# group normalization is better than batch normalization for small batch sizes, and is used in VAE. Inside each sample, the channels are divided into groups, and the mean and variance are computed for each group. The number of groups is usually set to 32, but can be adjusted based on the number of channels.
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

# double spatial resolution upsampling and downsampling. The upsampling is done by nearest neighbor interpolation followed by a convolution, and the downsampling is done by a convolution with stride 2. The kernel size of the convolution is 3, and the padding is 1 for upsampling and 0 for downsampling.
class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) # a convolution with kernel size 3 and padding 1 preserves the spatial resolution of the input, while the stride of 1 ensures that the output has the same number of channels as the input. The convolution is applied after upsampling to learn a better representation of the upsampled features.
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

# The downsampling also pads the input with zeros to make the output size half of the input size. why not use pooling? because pooling loses information, while convolution can learn to preserve important features. The padding is done by adding a row and column of zeros to the bottom and right of the input, so that the output size is exactly half of the input size.
class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0) # a convolution with stride 2 reduces the spatial resolution by half, and the kernel size of 3 allows the convolution to capture local features in the input. 
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout): # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        # the path is GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Dropout -> Conv3x3
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # shortcut connection: if in_channels != out_channels, use a 1x1 convolution to match the channels; otherwise, use identity.
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        # output = original input + transformed input (residual connection)
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module): # let every pixel communicate globally with every other pixel in the same feature map
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        self.norm = Normalize(in_channels)
        # initialize a convolutional layer to compute query, key, and value from the input. The output channels are 3 times the input channels, since we need to compute q, k, v for attention.
        self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
        # compute the scaling factor for the attention weights. The scaling factor is 1/sqrt(C), where C is the number of channels. This is a common practice in attention mechanisms to prevent the dot products from growing too large.
        self.w_ratio = int(in_channels) ** (-0.5)
        # initialize a convolutional layer to project the output of the attention mechanism back to the original number of channels. The kernel size is 1, so it only mixes the channels without changing the spatial dimensions.
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        qkv = self.qkv(self.norm(x)) # input shape is B,C,H,W
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1) # split into q, k, v along the channel dimension
        
        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j] -> how much location i attends to location j
        w = F.softmax(w, dim=2)
        
        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h) # original input + transformed input (residual connection)


def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    """
    1. Why there exist conv_in and conv_out?
        The raw image has only 3 channels, but ResNet blocks expect a richer feature space (128) -> image space to feature space
        The output of the last ResNet block has 1024 channels, but the compact latent space has only 256 channels -> feature space to latent space.
    2. Why 2 residual blocks per resolution level?
        1 block is not enough to learn a good representation, and 3 blocks are too many and will increase the model size and training time. So 2 blocks is a good balance.
    3. Why downsample at the end of each resolution level except the last one?
        The encoder's job is to compress the input image into a compact latent representation. Downsampling reduces the spatial resolution of the feature maps, which helps to achieve this compression. However, we don't want to downsample the last resolution level because we want to preserve as much information as possible in the latent representation.
    4. Why add self-attention layer at the last level?
        Use attention only after the feature map is small enough to avoid high computational cost.
    5. What is the effect of the middle block?
        The middle block is a bottleneck that allows the model to learn more complex representations by adding attention and residual connections. The attention inside the middle block is useful because the tensor is small enough for global communication.
    """
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch # base channel number
        self.num_resolutions = len(ch_mult) # channel multiplier for each resolution, e.g., (1, 2, 4, 8) means the first resolution has 128 channels, the second has 2*128=256 channels, etc.
        self.downsample_ratio = 2 ** (self.num_resolutions - 1) # 2 ** (4-1) = 8, if the input image is 256x256, the output will be 32x32 -> latent map of VQVAE is 32x32
        self.num_res_blocks = num_res_blocks # number of residual blocks per resolution is 2
        self.in_channels = in_channels # number of input channels, e.g., 3 for RGB images
        
        # convert the first convolutional layer that maps the input image to the base number of channels. B, 3, H, W -> B, 128, H, W
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1) 
        
        in_ch_mult = (1,) + tuple(ch_mult) # (1, 1, 2, 4, 8) for the input channels of each resolution level
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions): # for each resolution level
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level] # the number of input channels for the current resolution level
            block_out = ch * ch_mult[i_level] # the number of output channels for the current resolution level
            # level 0: input 128 -> output 128
            # level 1: input 128 -> output 256
            # level 2: input 256 -> output 512
            # level 3: input 512 -> output 1024
            for i_block in range(self.num_res_blocks): # 2 blocks
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                # each level has 2 residual blocks, the first block takes block_in channels and outputs block_out channels, the second block takes block_out channels and outputs block_out channels. So after the first block, we need to update block_in to be block_out for the next block.
                # ResBlock 128 -> 256 then 256 -> 256
                block_in = block_out
                # add self-attention layer at the last level
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            # downsample at the end of each level except the last one, so that the output of the last level is not downsampled. The downsampling is done by a convolution with stride 2, which reduces the spatial resolution by half. The number of channels remains the same after downsampling.
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        
        # middle that shape will remain the same
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end that maps the output of the last resolution level (maybe B, block_in, H/8, W/8) to the latent space (B, z_channels, H/8, W/8).
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)

    # Full flow for encoder:
    # input image: B, 3, H, W
    # conv_in:
    #     B, 3, H, W -> B, 128, H, W
    # level 0:
    #     ResBlock 128 -> 128
    #     ResBlock 128 -> 128
    #     -> B, 128, H, W
    # level 1:
    #     ResBlock 128 -> 256
    #     ResBlock 256 -> 256
    #     -> B, 256, H/2, W/2
    # level 2:
    #     ResBlock 256 -> 512
    #     ResBlock 512 -> 512
    #     -> B, 512, H/4, W/4
    # level 3:
    #     ResBlock 512 -> 1024
    #     Attention
    #     ResBlock 1024 -> 1024
    #     Attention
    #     -> B, 1024, H/8, W/8
    # middle:
    #     ResBlock 1024 -> 1024
    #     Attention
    #     ResBlock 1024 -> 1024
    #     -> B, 1024, H/8, W/8
    # conv_out:
    #     B, 1024, H/8, W/8 -> B, z_channels, H/8, W/8

# For default ch_mult=(1,2,4,8): output is B,z_channels,H/8,W/8.
# For VAR VQVAE ch_mult=(1,1,2,2,4): output is B,z_channels,H/16,W/16.
    
    def forward(self, x):
        # B,3,H,W -> B,z_channels,H/8,W/8 as z_channels is the number of channels in the latent space.
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h



class Decoder(nn.Module): # compressed latent feature map -> reconstructed image
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult) # 4
        self.num_res_blocks = num_res_blocks # 2
        self.in_channels = in_channels
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1] # the number of channels at the lowest resolution level, which is the output of the last downsampling layer in the encoder. This will be the input to the decoder. Here is 128 * 8 = 1024
        
        # z to block_in to convert (B, z_channels, H/16, W/16) to (B, block_in, H/16, W/16). The latent space has z_channels channels, but the decoder expects block_in channels as input. So we need a convolutional layer to map from z_channels to block_in channels.
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle does not change the shape but add more latent representation before upsampling.
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout) # ResBlock 1024 -> 1024
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa) # Attention 1024 -> 1024
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout) # ResBlock 1024 -> 1024
        
        # upsampling are built in reverse order of downsampling, so that the output of the last upsampling layer has the same spatial resolution as the input image.
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): # spatial resolution: 32x32 -> 64x64 -> 128x128 -> 256x256 and channel depth: 1024 -> 512 -> 256 -> 128
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1): # decoder has 3 blocks per resolution level because generation/reconstruction is more difficult than compression, so we need more capacity to learn the mapping from latent space to image space.
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa: # add self-attention layer at the first level when decoder == last level when encoder. The reason we explain above, as smallest spatial resolution, we can afford the computational cost of self-attention.
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0: # upsample at the end of each level except the last one.
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # insert at the beginning of the list so that the first level is at index 0, and the last level is at index num_resolutions-1. Let the forward still runs .... ? hard to understand
        
        # end convert (B, 128, H, W) to (B, 3, H, W) to reconstruct the image. 
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    # Full flow for decoder:
    # z_channels = 32 -> input latent = B, 32, 32, 32
    # conv_in:
    #     B, 32, 32, 32 -> B, 1024, 32, 32

    # middle:
    #     ResBlock 1024
    #     Attention
    #     ResBlock 1024
    #     -> B, 1024, 32, 32 (B, out_channels, H/8, W/8)

    # level 3:
    #     ResBlock 1024 -> 1024
    #     Attention
    #     ResBlock 1024 -> 1024
    #     Attention
    #     ResBlock 1024 -> 1024
    #     Attention
    #     Upsample
    #     -> B, 1024, 64, 64 (B, out_channels, H/4, W/4)

    # level 2:
    #     ResBlock 1024 -> 512
    #     ResBlock 512 -> 512
    #     ResBlock 512 -> 512
    #     Upsample
    #     -> B, 512, 128, 128 (B, out_channels/2, H/2, W/2)

    # level 1:
    #     ResBlock 512 -> 256
    #     ResBlock 256 -> 256
    #     ResBlock 256 -> 256
    #     Upsample
    #     -> B, 256, 256, 256 (B, out_channels/4, H, W)

    # level 0:
    #     ResBlock 256 -> 128
    #     ResBlock 128 -> 128
    #     ResBlock 128 -> 128
    #     -> B, 128, 256, 256 (B, out_channels/8, H/2, W/2)

    # conv_out:
    #     B, 128, 256, 256 -> B, 3, 256, 256
    
    def forward(self, z):
        # B,z_channels,H/8,W/8 -> B,3,H,W
        """
        z -> conv_in -> middle -> upsampling

        for each level from small to large:
            apply 3 ResBlocks
            if this low-res level has attention, apply attention
            if not final level, double H and W 
        """
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
