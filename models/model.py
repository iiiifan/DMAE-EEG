import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

class EEGTransformerEncoder(nn.Module):
    """脑电信号专用的Transformer编码器"""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            EEGTransformerBlock(dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GraphAwarePooling(nn.Module):
    """脑区感知的空间池化（封装为模块）"""

    def __init__(self):
        super().__init__()

    def forward(self, x, group_indices, adj_matrix=None):
        batch_size, C, T = x.shape
        pooled = torch.zeros(batch_size, len(group_indices), T, device=x.device)

        adj_matrix = torch.ones(C, C, device=x.device) if adj_matrix is None else adj_matrix

        for m, indices in enumerate(group_indices):
            region_x = x[:, indices, :]
            region_adj = adj_matrix[indices][:, indices]

            weights = region_adj.sum(dim=1, keepdim=True)
            weights = weights / weights.sum()

            pooled[:, m] = torch.einsum('bct,cd->bdt', region_x, weights).squeeze()
        return pooled


class SpatioTemporalPatch(nn.Module):
    """时空块生成（封装为模块）"""

    def __init__(self, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer

    def forward(self, signals, group_indices):
        batch_size, C, T = signals.shape
        pooled = self.pool_layer(signals, group_indices)

        time_blocks = []
        block_size = T // 4
        for i in range(4):
            start = i * block_size
            end = (i + 1) * block_size if i < 3 else T
            time_blocks.append(pooled[..., start:end])

        return torch.cat(time_blocks, dim=1)

class EEGTransformerBlock(nn.Module):
    """脑电Transformer基础模块"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = EEGAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ffn = EEGFeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class EEGAttention(nn.Module):
    """脑电信号专用的注意力机制"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class EEGFeedForward(nn.Module):
    """脑电信号专用的前馈网络"""

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class EEGDenoisingMAE(nn.Module):
    """基于MAE的脑电信号去噪模型"""

    def __init__(self,
                 dataset_names,
                 num_patches: int,
                 vit_dim: int,
                 vit_depth: int,
                 vit_heads: int,
                 vit_mlp_dim: int,
                 masking_ratio: float = 0.75,
                 decoder_dim: int = 512,
                 decoder_depth: int = 2,
                 temporal_dim: int = 100,
                 spatial_groups: int = 10,
                 device: torch.device = 'cpu'):
        super().__init__()
        self.device = device
        self.masking_ratio = min(max(masking_ratio, 0), 0.9)
        self.spatial_groups = spatial_groups
        self.temporal_dim = temporal_dim

        self.input_norm = nn.LayerNorm(vit_dim)
        self.decoder_norm = nn.LayerNorm(decoder_dim)

        # 编码器组件
        self.patch_embed = nn.Linear(temporal_dim, vit_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, vit_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_dim))
        self.encoder = EEGTransformerEncoder(
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            dim_head=vit_dim // vit_heads,
            mlp_dim=vit_mlp_dim
        )

        # 解码器组件
        self._init_decoder(vit_dim, decoder_dim, decoder_depth)

        # 投影层
        self.patch_projector = nn.Linear(decoder_dim, temporal_dim)


        # 新增可学习的损失缩放参数
        self.loss_scales = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1, device=device))
            for name in dataset_names
        })

        self.graph_pool = GraphAwarePooling()
        self.st_patch = SpatioTemporalPatch(self.graph_pool)


    def _init_decoder(self, vit_dim, dim, depth):
        """初始化解码器"""
        self.encoder_to_decoder = nn.Linear(vit_dim, dim) if vit_dim != dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.decoder = EEGTransformerEncoder(
            dim=dim,
            depth=depth,
            heads=8,  # 固定解码器头数
            dim_head=dim // 8,
            mlp_dim=dim * 4
        )
        self.pos_embed_decoder = nn.Embedding(self.spatial_groups * 4, dim)

    def _graph_aware_pooling(self, x, group_indices, adj_matrix=None):
        """脑区感知的空间池化"""
        batch_size, C, T = x.shape
        pooled = torch.zeros(batch_size, len(group_indices), T, device=self.device)

        adj_matrix = torch.ones(C, C, device=self.device) if adj_matrix is None else adj_matrix.to(self.device)

        for m, indices in enumerate(group_indices):
            region_x = x[:, indices, :]
            region_adj = adj_matrix[indices][:, indices]

            weights = region_adj.sum(dim=1, keepdim=True)
            weights = weights / weights.sum()

            pooled[:, m] = torch.einsum('bct,cd->bdt', region_x, weights).squeeze()
        return pooled

    def _create_spatiotemporal_patches(self, signals, group_indices):
        """生成时空联合特征块"""
        batch_size, C, T = signals.shape
        # signals = signals.permute(0, 3, 2, 1).squeeze(-1)  # [N,C,T]

        # 空间池化
        pooled = self._graph_aware_pooling(signals, group_indices)  # [B, 10, T]

        # 时间分块
        time_blocks = []
        block_size = T // 4
        for i in range(4):
            start = i * block_size
            end = (i + 1) * block_size if i < 3 else T
            time_blocks.append(pooled[..., start:end])

        patches = torch.cat(time_blocks, dim=1)  # [B, 40, block_size]
        return patches.to(self.device)

    # 以下为新增辅助方法
    def _name_to_index(self, name):
        """将数据集名称映射为参数索引"""
        return list(self.loss_scales.keys()).index(name)

    @property
    def index_to_name(self):
        """反向映射字典"""
        return {i: name for i, name in enumerate(self.loss_scales.keys())}


    def forward(self, noisy_signals, clean_signals, group_index, dataset_ids=None):
        # 生成时空块 [B, num_patches, temporal_dim]
        # noisy_patches = self._create_spatiotemporal_patches(noisy_signals, group_index)
        # clean_patches = self._create_spatiotemporal_patches(clean_signals, group_index)
        noisy_patches = self.st_patch(noisy_signals, group_index)
        clean_patches = self.st_patch(clean_signals, group_index)
        # 掩码处理
        batch_size, num_patches, _ = noisy_patches.shape
        num_masked = int(num_patches * self.masking_ratio)

        visible_emb = self.patch_embed(noisy_patches)
        visible_emb = self.input_norm(visible_emb)  # 新增层归一化
        visible_emb += self.pos_embedding[:, 1:num_patches+1]

        # 随机选择掩码位置
        shuffle_idx = torch.rand(batch_size, num_patches, device=self.device).argsort(dim=-1)
        mask_idx, visible_idx = shuffle_idx[:, :num_masked], shuffle_idx[:, num_masked:]

        # 编码可见块
        visible_emb = self.patch_embed(noisy_patches) + self.pos_embedding[:, 1:num_patches + 1]
        encoded = self.encoder(visible_emb.gather(1, visible_idx.unsqueeze(-1).expand(-1, -1, visible_emb.size(-1))))

        # 解码器输入
        decoder_input = self.encoder_to_decoder(encoded)
        decoder_input = self.decoder_norm(decoder_input)  # 新增层归一化
        mask_tokens = self.mask_token + self.pos_embed_decoder(mask_idx)
        decoder_input = torch.cat([mask_tokens, decoder_input], dim=1)

        # 重构预测
        reconstructed = self.decoder(decoder_input)
        pred_patches = self.patch_projector(reconstructed[:, :num_masked])

        # 计算损失
        target_patches = clean_patches.gather(1, mask_idx.unsqueeze(-1).expand(-1, -1, self.temporal_dim))
        per_sample_loss = F.mse_loss(pred_patches, target_patches, reduction='none')
        per_sample_loss = per_sample_loss.mean(dim=[1, 2])  # 假设shape为[B, patches, temporal_dim]
        # 应用缩放因子
        if dataset_ids is not None:
            # 确保dataset_ids是张量
            if not isinstance(dataset_ids, torch.Tensor):
                dataset_ids = torch.tensor([self._name_to_index(name) for name in dataset_ids],
                                           device=per_sample_loss.device)
            # 按数据集分组计算缩放后的损失
            unique_ids = torch.unique(dataset_ids)
            scaled_loss = 0.0
            for uid in unique_ids:
                mask = (dataset_ids == uid)
                scale = self.loss_scales[self.index_to_name[uid.item()]]
                scaled_loss += (per_sample_loss[mask] * scale).sum()
            loss = scaled_loss / batch_size
        else:
            loss = per_sample_loss.mean()

        return loss

    def get_loss_scales(self):
        """获取当前损失缩放因子"""
        return {name: param.item() for name, param in self.loss_scales.named_parameters()}

    def print_loss_scales(self):
        """打印当前缩放因子"""
        scales = self.get_loss_scales()
        print("当前损失缩放因子:")
        for name, value in scales.items():
            print(f"  {name}: {value:.4f}")
