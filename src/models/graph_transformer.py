"""
Spatio-Temporal Graph Transformer Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalGraphTransformer(nn.Module):
    """
    Graph Transformer combining temporal dynamics and spatial relationships
    for power grid failure prediction.
    """
    
    def __init__(self, seq_dim: int, static_dim: int, num_nodes: int,
                 emb_dim: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            seq_dim: Dimension of sequential features
            static_dim: Dimension of static features
            num_nodes: Number of nodes in the graph
            emb_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Time encoding
        self.time_enc = nn.Linear(1, emb_dim)
        
        # Node embeddings
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        
        # Input projection for sequence features
        self.input_proj = nn.Linear(seq_dim, emb_dim)
        
        # Temporal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.temporal_enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Spatial Graph Attention
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # Spatial features processing
        self.spatial_norm = nn.LayerNorm(emb_dim)
        self.spatial_dropout = nn.Dropout(dropout)
        
        # Static feature projection
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.BatchNorm1d(emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion and output head
        self.out_proj = nn.Sequential(
            nn.Linear(emb_dim + emb_dim // 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.BatchNorm1d(emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, seq: torch.Tensor, static: torch.Tensor, 
                node_idx: torch.Tensor, adj_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            seq: Sequential features [B, T, F]
            static: Static features [B, F_static]
            node_idx: Node indices [B]
            adj_mask: Adjacency mask [B, B]
            
        Returns:
            logits: Output logits [B]
        """
        B, T, F = seq.shape
        
        # Project inputs
        x = self.input_proj(seq)  # [B, T, emb]
        
        # Add time encoding
        time_idx = torch.arange(T, device=seq.device).unsqueeze(-1).float() / T
        t_enc = self.time_enc(time_idx)  # [T, emb]
        x = x + t_enc.unsqueeze(0)
        
        # Temporal encoding
        x = self.temporal_enc(x)  # [B, T, emb]
        h_t = x[:, -1, :]         # [B, emb] - take last time step
        
        # Add node embedding
        node_e = self.node_emb(node_idx)  # [B, emb]
        node_feats = h_t + node_e         # [B, emb]
        
        # Spatial attention across nodes
        q = k = v = node_feats.unsqueeze(1)  # [B, 1, emb]
        
        if adj_mask is not None:
            # Apply spatial attention with adjacency constraints
            attn_mask = ~adj_mask.bool()  # True = don't attend
            spatial_out, _ = self.spatial_attn(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                attn_mask=attn_mask
            )
            spatial_out = spatial_out.transpose(0, 1)
        else:
            # No adjacency constraints
            spatial_out, _ = self.spatial_attn(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1)
            )
            spatial_out = spatial_out.transpose(0, 1)
        
        # Process spatial features
        out_sp = spatial_out.squeeze(1)  # [B, emb]
        out_sp = self.spatial_norm(out_sp)
        out_sp = self.spatial_dropout(out_sp)
        
        # Static features
        st = self.static_proj(static)  # [B, emb//2]
        
        # Fuse and predict
        combined = torch.cat([out_sp, st], dim=1)
        logits = self.out_proj(combined).squeeze(1)
        
        return logits