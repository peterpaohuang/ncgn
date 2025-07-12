# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from functools import partial
from na2d import NeighborhoodAttention2D, NeighborhoodAttention2D
from flex_patch_embed import FlexiPatchEmbed, resize_abs_pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTDMPBlock(nn.Module):
    """
    A DiT block with dynamic attention size
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = NeighborhoodAttention2D(hidden_size, num_heads=num_heads, kernel_size=2, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, k_neighbors):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), k_neighbors)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT DMP.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

def find_nearest_value(n):
    if n <= 160:      # Midpoint between 64 and 256
        return 64
    elif n <= 640:    # Midpoint between 256 and 1024
        return 256
    else:
        return 1024

def adaptive_schedule(t, curvature=10):
    """
    Borrows from exponential schedule. Currently matching complexity of original DiT with patch_size=4

    We exponentially interpolate between patch_size=8 at highest noise to patch_size=1 at lowest noise
    The k_neighbors are determined based on patch_size to match complexity of DiT with patch_size=4

    patch_size = [1,2,4]
    k_neighbor = [4, 16, 64], determined by k_neighbor = 4096 / (32 / patch_size)^2
    runtime: O(p=4) -> O((32/4)^4 = 4096)
    """
    t = (1000 - t) / 1000
    start_clusters = 64 # assumes patch_size = 8, change for different patch size
    end_clusters = 1024 # assumes patch_size = 1, change for different patch size
    n_clusters = (end_clusters - start_clusters) * ((math.exp(curvature * t) - 1)/(math.exp(curvature) - 1)) + start_clusters
    rounded_n_clusters = find_nearest_value(n_clusters) # round to nearest n_clusters = [1024, 256, 64]
    patch_size = math.sqrt(1024 / rounded_n_clusters) # get patch_size = [1,2,4]
    k_neighbors = 4096 / (32 / patch_size)**2

    return int(patch_size), int(k_neighbors)


class DiT_DMP(nn.Module):
    """
    Dynamic Message Passing implementation of a Diffusion model with a Transformer backbone.

    At highest nosie, patch_size = 8 and k_neighbors = 256
    At lowest noise, patch_size = 1 and k_neighbors = 4

    Matches time complexity of original DiT with constant patch_size = 4 and full attention.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        patch_size = 8 # set to highest patch_size for initializaiton
        grid_size = 4
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.img_size = (input_size, input_size)

        self.x_embedder = FlexiPatchEmbed(input_size, patch_size, grid_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTDMPBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_ps_1 = FinalLayer(hidden_size, 1, self.out_channels)
        self.final_layer_ps_2 = FinalLayer(hidden_size, 2, self.out_channels)
        self.final_layer_ps_4 = FinalLayer(hidden_size, 4, self.out_channels)

        # Position embedding resizing function
        self.resize_pos_embed = partial(
            resize_abs_pos_embed,
            old_size=(grid_size,grid_size),
            interpolation="bicubic",
            antialias=True,
            num_prefix_tokens=0,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_ps_1.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_ps_1.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_ps_1.linear.weight, 0)
        nn.init.constant_(self.final_layer_ps_1.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_ps_2.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_ps_2.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_ps_2.linear.weight, 0)
        nn.init.constant_(self.final_layer_ps_2.linear.bias, 0)

        nn.init.constant_(self.final_layer_ps_4.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer_ps_4.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_ps_4.linear.weight, 0)
        nn.init.constant_(self.final_layer_ps_4.linear.bias, 0)

    def unpatchify(self, x, patch_size):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def _pos_embed(self, x, patch_size):
        # Resize position embedding based on current patch size
        new_size = (
            int(self.img_size[0] // patch_size[0]),
            int(self.img_size[1] // patch_size[1]),
        )
        pos_embed = self.resize_pos_embed(self.pos_embed, new_size)

        x = x + pos_embed
        return x
    
    def forward(self, x, t, y):
        """
        Forward pass of DiT DMP.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        ps, k_neighbors = adaptive_schedule(t[0])
        # ps = 4
        # k_neighbors = 64
        x, ps = self.x_embedder(x, ps, return_patch_size=True)   # (N, T, D), where T = H * W / patch_size ** 2
        x = self._pos_embed(x, ps)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c, k_neighbors)                      # (N, T, D)
        
        if ps[0] == 1:
            x = self.final_layer_ps_1(x, c)                # (N, T, patch_size ** 2 * out_channels)
        elif ps[0] == 2:
            x = self.final_layer_ps_2(x, c)                # (N, T, patch_size ** 2 * out_channels)
        elif ps[0] == 4:
            x = self.final_layer_ps_4(x, c)                # (N, T, patch_size ** 2 * out_channels)

        x = self.unpatchify(x, ps)                   # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT DMPConfigs                              #
#################################################################################

def DiT_DMP_XL_4(**kwargs):
    return DiT_DMP(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_DMP_L_4(**kwargs):
    return DiT_DMP(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_DMP_B_4(**kwargs):
    return DiT_DMP(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_DMP_S_4(**kwargs):
    return DiT_DMP(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


DiT_dmp_models = {
    'DiT-DMP-XL/4': DiT_DMP_XL_4, 'DiT-DMP-L/4':  DiT_DMP_L_4,
    'DiT-DMP-B/4':  DiT_DMP_B_4, 'DiT-DMP-S/4':  DiT_DMP_S_4,
}
