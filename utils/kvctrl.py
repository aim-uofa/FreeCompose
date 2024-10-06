import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .attention import AttentionBase

from torchvision.utils import save_image

class KVReplace(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = 16
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("KV replace at denoising steps: ", self.step_idx)
        print("KV replace at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        
        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)
        # qu_s, qc_s, qu_t, qc_t = q.chunk(4)
        # ku_s, kc_s, ku_t, kc_t = k.chunk(4)
        # vu_s, vc_s, vu_t, vc_t = v.chunk(4)
        # attnu_s, attnc_s, attnu_t, attnc_t = attn.chunk(4)

        # source image branch
        out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)

        # target image branch, concatenating source and target [K, V]
        # out_u_t = self.attn_batch(qu_t, torch.cat([ku_s, ku_t]), torch.cat([vu_s, vu_t]), sim[:num_heads], attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        # out_c_t = self.attn_batch(qc_t, torch.cat([kc_s, kc_t]), torch.cat([vc_s, vc_t]), sim[:num_heads], attnc_t, is_cross, place_in_unet, num_heads, **kwargs)
        out_u_t = self.attn_batch(qu_t, ku_s, vu_s, sim, attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        # out_u_t = super().forward(qu_t, ku_t, vu_t, sim, attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
        out_c_t = self.attn_batch(qc_t, kc_s, vc_s, sim, attnc_t, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out
    
class KVSelfReplace(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, mask=None, threshold=0.1):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            model_type: the model type, SD or SDXL
        """
        super().__init__()
        self.total_steps = total_steps
        self.total_layers = 16
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        
        mask = torch.from_numpy(mask).float()
        mask = mask.reshape(1, 1, mask.shape[0], mask.shape[1])
        self.mask = mask
        if self.mask is not None:
            self.mask_list = {}
            for i in [64, 32, 16, 8]:
                mask = F.interpolate(self.mask, size=(i, i), mode='bilinear', align_corners=False)
                mask = mask.reshape(-1)
                mask = (mask > threshold).to(dtype=torch.bool)
                self.mask_list[i**2] = mask
                print(f"Mask shape: {i}x{i}, {torch.count_nonzero(mask)}")
        
        print("KV self replace at denoising steps: ", self.step_idx)
        print("KV self replace at U-Net layers: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Performing attention for a batch of queries, keys, and values
        """
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def random_replace(self, input_tensor):
        n = input_tensor.shape[1]
        # mask = ~self.mask_list[n]
        # # random choice from input tensor
        # idx = torch.randperm(n)
        # input_tensor[:, mask] = input_tensor[:, idx[mask]]
        return input_tensor[:, ~self.mask_list[n]]
        return input_tensor

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        
        qu_s, qu_t, qc_s, qc_t = q.chunk(4)
        ku_s, ku_t, kc_s, kc_t = k.chunk(4)
        vu_s, vu_t, vc_s, vc_t = v.chunk(4)
        attnu_s, attnu_t, attnc_s, attnc_t = attn.chunk(4)
        # qu_s, qc_s, qu_t, qc_t = q.chunk(4)
        # ku_s, kc_s, ku_t, kc_t = k.chunk(4)
        # vu_s, vc_s, vu_t, vc_t = v.chunk(4)
        # attnu_s, attnc_s, attnu_t, attnc_t = attn.chunk(4)

        # source image branch
        if self.mask is not None:
            out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)
            # out_u_s = self.attn_batch(qu_s, self.random_replace(ku_s), self.random_replace(vu_s), sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
            # out_c_s = self.attn_batch(qc_s, self.random_replace(kc_s), self.random_replace(vc_s), sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)
            
            # target image branch, concatenating source and target [K, V]
            out_u_t = self.attn_batch(qu_t, self.random_replace(ku_t), self.random_replace(vu_t), sim, attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_t = self.attn_batch(qc_t, self.random_replace(kc_t), self.random_replace(vc_t), sim, attnc_t, is_cross, place_in_unet, num_heads, **kwargs)
            # out_u_t = super().forward(qu_t, ku_t, vu_t, sim, attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
            # out_c_t = super().forward(qc_t, kc_t, vc_t, sim, attnc_t, is_cross, place_in_unet, num_heads, **kwargs)
        else:
            out_u_s = super().forward(qu_s, ku_s, vu_s, sim, attnu_s, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_s = super().forward(qc_s, kc_s, vc_s, sim, attnc_s, is_cross, place_in_unet, num_heads, **kwargs)

            # target image branch, concatenating source and target [K, V]
            out_u_t = super().forward(qu_t, ku_t, vu_t, sim, attnu_t, is_cross, place_in_unet, num_heads, **kwargs)
            out_c_t = super().forward(qc_t, kc_t, vc_t, sim, attnc_t, is_cross, place_in_unet, num_heads, **kwargs)

        out = torch.cat([out_u_s, out_u_t, out_c_s, out_c_t], dim=0)

        return out