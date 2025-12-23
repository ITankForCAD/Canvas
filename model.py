import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


class BaseModule(nn.Module):
    """
    Helper to handle custom layer weight initialization
        1. Transformer relevant layers sample from 
        xavier uniform for better convergence
        2. Norms and biases sample from constant
    """
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight) # init the correct wrapped layers


class ResidualBlock1D(nn.Module):
    """
    Helper for residual signals in grid creation.
    """
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )
    
    def forward(self, x):
        return x + self.net(x)


class MachineLanguageEncoder(BaseModule):
    """
    Compresses text embeddings --> Dense latent grids
    """
    def __init__(self, hp):
        super().__init__()
        self.hidden = hp["hidden size"]
        self.latent = hp["latent channels"]
        self.params = hp

        self.input_norm = nn.LayerNorm(self.hidden)
        self.anchor_net = nn.Sequential(
            nn.Conv1d(self.hidden, self.latent, kernel_size=3, padding=1),
            ResidualBlock1D(self.latent),
            ResidualBlock1D(self.latent),
            ResidualBlock1D(self.latent)
        )
        self.context_proj = nn.Linear(self.hidden, self.latent)
        layer = nn.TransformerDecoderLayer(
            d_model=self.latent,
            nhead=hp["encoder heads"],
            dim_feedforward=self.latent * 4,
            batch_first=True,
            norm_first=True,
            activation=F.tanh # better negative range
        )
        self.refiner = nn.TransformerDecoder(layer, num_layers=hp["encoder layers"])
        self.viz = nn.Conv2d(self.latent, hp["viz channels"], kernel_size=1)
        self.apply(self._init_weights)

    def forward(self, embeddings, attention_mask=None):
        B, L, D = embeddings.shape
        x = self.input_norm(embeddings)

        pixels = self.params["grid height"] * self.params["grid width"]
        remainder = L % pixels
        if remainder != 0:
            pad_len = pixels - remainder
            x = torch.cat([x, torch.zeros(B, pad_len, D, device=x.device)], dim=1)
            if attention_mask is not None:
                pad_mask = torch.zeros(B, pad_len, device=x.device)
                attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        
        x_t = x.transpose(1, 2)
        anchor = self.anchor_net(x_t).transpose(1, 2)
        memory = self.context_proj(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (1.0 - attention_mask).bool()
        
        latents = self.refiner(
            tgt=anchor,
            memory=memory,
            memory_key_padding_mask=key_padding_mask
        )
        latents += anchor
        latents = latents.transpose(1, 2)

        B, D, L_latent = latents.shape
        num_grids = L_latent // pixels
        grids = latents.view(B, -1, num_grids, self.params["grid height"], self.params["grid width"])
        machine_grids = grids.permute(0,2,1,3,4).reshape(B * num_grids, -1, self.params["grid height"], self.params["grid width"])
        human_grids = self.viz(machine_grids)
        return machine_grids, human_grids

class MachineLanguageDecoder(BaseModule):
    """
    Dense latent grids --> Reconstructs text embeddings
    """
    def __init__(self, hp):
        super().__init__()
        self.hidden = hp["hidden size"]
        self.latent = hp["latent channels"]
        self.params = hp

        self.anchor_net = nn.Sequential(
            nn.Conv1d(self.latent, self.hidden, kernel_size=3, padding=1),
            ResidualBlock1D(self.hidden),
            ResidualBlock1D(self.hidden),
            ResidualBlock1D(self.hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, 
            nhead=hp["decoder heads"], 
            dim_feedforward=self.hidden * 2,
            batch_first=True,
            norm_first=True,
            activation=F.tanh
        )
        self.refiner = nn.TransformerEncoder(layer, num_layers=hp["decoder layers"], enable_nested_tensor=False)
        
        self.apply(self._init_weights)

    def forward(self, latents):
        BN, C, H, W = latents.shape
        x = latents.view(BN, C, -1)
        anchor = self.anchor_net(x)
        anchor = anchor.transpose(1, 2)
        refined = self.refiner(anchor)
        recon = anchor + refined
        return recon

class ContextAE(nn.Module):
    """
    Masked, Cross-Modal Autoencoder for Data Compression using Learned
    Visual Representations of Text Embeddings
    """
    def __init__(self, encoder, decoder, mask_ratio=0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        B, L, D = inputs_embeds.shape
        x_input = inputs_embeds
        if self.training and self.mask_ratio > 0:
            probs = torch.rand(B, L, device=inputs_embeds.device)
            mask = probs < self.mask_ratio
            mask[:, 0] = False 
            keep_mask = (~mask).unsqueeze(-1).to(dtype=inputs_embeds.dtype)
            x_input = inputs_embeds * keep_mask
        
        machine_grids, human_grids = self.encoder(x_input, attention_mask)
        raw_recon = self.decoder(machine_grids)
        reconstructed = raw_recon.reshape(B, -1, D)

        if reconstructed.shape[1] > L:
            reconstructed = reconstructed[:, :L, :]
        elif reconstructed.shape[1] < L:
            diff = L - reconstructed.shape[1]
            pad = torch.zeros(B, diff, D, device=reconstructed.device)
            reconstructed = torch.cat([reconstructed, pad], dim=1)

        if self.training:
            mse = self.mse_loss(reconstructed, inputs_embeds, attention_mask)
            nce = self.info_nce_loss(reconstructed, inputs_embeds, attention_mask)
            cos = self.cosine_loss(reconstructed, inputs_embeds, attention_mask)
            loss = cos + (0.1 * mse) + (0.2 * nce)

            return {
                "loss": loss,
                "mse": mse,
                "nce": nce,
                "cos": cos,
                "logits": reconstructed
            }
        else:
            return {
                "logits": reconstructed,
                "latents": machine_grids,
                "human_grids": human_grids
            }
    
    def cosine_loss(self, pred, target, mask):
        mask_bool = mask.view(-1).bool()
        p_flat = pred.reshape(-1, pred.shape[-1])[mask_bool]
        t_flat = target.reshape(-1, target.shape[-1])[mask_bool]
        cosine_target = torch.ones(p_flat.shape[0], device=p_flat.device)
        loss = F.cosine_embedding_loss(p_flat, t_flat, cosine_target, reduction="mean")
        return loss

    def mse_loss(self, pred, target, mask):
        min_len = min(pred.shape[1], target.shape[1])
        p = pred[:, :min_len]
        t = target[:, :min_len]
        m = mask[:, :min_len]
        diff = (p - t) ** 2
        loss_per_token = diff.mean(dim=-1)
        m_float = m.to(dtype=loss_per_token.dtype)
        return (loss_per_token * m_float).sum() / (m_float.sum() + 1e-8)

    def info_nce_loss(self, pred, target, mask, temperature=0.07):
        mask_bool = mask.view(-1).bool()
        p_flat = pred.reshape(-1, pred.shape[-1])[mask_bool]
        t_flat = target.reshape(-1, target.shape[-1])[mask_bool]
        if p_flat.shape[0] > 2048:
            indices = torch.randperm(p_flat.shape[0], device=pred.device)[:2048]
            p_flat = p_flat[indices]
            t_flat = t_flat[indices]
        
        p_norm = F.normalize(p_flat, dim=-1)
        t_norm = F.normalize(t_flat, dim=-1)
        logits = torch.mm(p_norm, t_norm.t()) / temperature
        labels = torch.arange(len(p_flat), device=logits.device)
        return F.cross_entropy(logits, labels)


class DirectTokenViT(nn.Module):
    """
    Feed precomputed grid tokens directly into ViT encoder,
    bypassing patch embedding.
    """

    def __init__(
        self,
        grid_dim=128,
        vit_name="google/vit-base-patch16-224",
        max_grids=512,
        freeze_vit=True
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained(vit_name)
        self.vit_dim = self.vit.config.hidden_size
        self.grid_proj = nn.Linear(grid_dim, self.vit_dim)
        self.cls_token = self.vit.embeddings.cls_token
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_grids + 1, self.vit_dim)
        )
        self.norm = self.vit.embeddings.layernorm
        if freeze_vit:
            for p in self.vit.encoder.parameters():
                p.requires_grad = False

    def forward(self, grid_tokens):
        B, N, _ = grid_tokens.shape
        assert N + 1 <= self.pos_embed.size(1)
        x = self.grid_proj(grid_tokens)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : N + 1]
        x = self.norm(x)
        out = self.vit.encoder(
            hidden_states=x,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True
        )
        tokens = out.last_hidden_state[:, 1:]
        return tokens

class QLayer(BaseModule):
    """
    Custom layer used in the Query Transformer.
    """
    def __init__(self, d_model, nhead, dim_ff):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, memory):
        attn_out, _ = self.cross_attn(
            query=queries,
            key=memory,
            value=memory
        )
        x = self.norm(queries + attn_out)
        x = x + self.ff(x)
        return x

class QFormer(BaseModule):
    """
    Query Transformer module used to expand the ViT
    vectors up to num_queries. (num_queries controls
    the token specific compression ratio)
    """
    def __init__(
        self,
        vit_dim,
        llm_dim,
        num_queries=8,
        num_layers=4,
        nhead=8,
        ff_mult=4
    ):
        super().__init__()

        self.num_queries = num_queries
        self.vit_proj = nn.Linear(vit_dim, llm_dim)

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_queries, llm_dim)
        )

        self.layers = nn.ModuleList([
            QLayer(
                d_model=llm_dim,
                nhead=nhead,
                dim_ff=llm_dim * ff_mult
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(llm_dim)

    def forward(self, vit_tokens):
        B, N, _ = vit_tokens.shape
        memory = self.vit_proj(vit_tokens)
        memory = memory.view(B * N, 1, -1)
        queries = self.query_tokens.expand(B * N, -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory)

        queries = self.final_norm(queries)
        queries = queries.view(B, N * self.num_queries, -1)
        return queries
