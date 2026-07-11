import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleSSM(nn.Module):
    """Egyszerűsített Mamba/SSM implementáció - STABIL"""
    def __init__(self, d_model, state_dim=16, conv_kernel=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Projektálások
        self.in_proj = nn.Linear(d_model, d_model * 2)  # A és B
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model  # Depthwise conv
        )
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Paraméterek
        # S4D inicializáció: A = -0.5 * (1 + i)
        A_init = -0.5 * (1 + torch.arange(state_dim).float())
        self.A = nn.Parameter(A_init)
        self.D = nn.Parameter(torch.randn(d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
        # Stabil inicializáció
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.02)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.D, mean=0.0, std=0.02)
    
    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        return: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # 1. Lineáris projektálás
        x_proj = self.in_proj(x)  # [batch, seq_len, 2*d_model]
        A_part, B_part = x_proj.chunk(2, dim=-1)
        
        # 2. Depthwise konvolúció
        x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # 3. SSM szimuláció (egyszerűsített)
        # Diszkrét A - STABILIZÁLVA
        delta = self.activation(A_part)  # [batch, seq_len, d_model]
        # Clamp a delta-t, hogy ne robbanjon fel az exp
        A_discrete = torch.exp(torch.clamp(self.A.unsqueeze(0).unsqueeze(0) * delta.unsqueeze(-1), max=10.0))
        A_discrete = A_discrete.view(batch, seq_len, self.d_model, self.state_dim)
        
        # B és C
        B = self.activation(B_part).unsqueeze(-1)  # [batch, seq_len, d_model, 1]
        C = x_conv.unsqueeze(-1)  # [batch, seq_len, d_model, 1]
        
        # 4. Scan (egyszerűsített)
        states = []
        current_state = torch.zeros(batch, self.d_model, self.state_dim, device=x.device)
        
        for t in range(seq_len):
           A_t = A_discrete[:, t]  # [batch, d_model, state_dim]
           B_t = B[:, t]  # [batch, d_model, 1]
           C_t = C[:, t]  # [batch, d_model, 1]
   
           # State update: x_t = A_t * x_{t-1} + B_t * u_t
           # B_t needs to be broadcasted to [batch, d_model, state_dim]
           current_state = A_t * current_state + B_t.expand(-1, -1, self.state_dim)
   
           # Output: y_t = C_t * x_t + D * u_t
           # Element-wise multiplication and sum reduction
           output_t = torch.sum(C_t.expand(-1, -1, self.state_dim) * current_state, dim=-1, keepdim=False)
           output_t = output_t + self.D * x[:, t]
   
           states.append(output_t.unsqueeze(1))
        x_ssm = torch.cat(states, dim=1)  # [batch, seq_len, d_model]
        
        # 5. Output projektálás
        x_out = self.out_proj(x_ssm)
        x_out = self.dropout(x_out)
        
        return x_out


class SimpleFlashAttention(nn.Module):
    """Egyszerűsített Flash Attention - STABIL (CPU/GPU kompatibilis)"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projektálások
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # Stabil inicializáció
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=0.02)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        mask: [batch, seq_len] vagy [batch, 1, seq_len, seq_len]
        """
        batch, seq_len, _ = x.shape
        
        # 1. QKV projektálás
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*d_model]
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. Scaled dot-product attention (stabil implementáció)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 3. Mask (ha van)
        if mask is not None:
            # Konvertálás bool-ra ha szükséges (True = maszkolás)
            if mask.dtype != torch.bool:
                mask = (mask == 0)
                
            # A maszk formátuma: True = maszkolás, False = megtartás
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len, seq_len]
            
            # Maszk alkalmazása (ahol True, ott -inf)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 4. Softmax (numerikusan stabil)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 5. Output
        attn_output = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_len, n_heads, head_dim]
        attn_output = attn_output.reshape(batch, seq_len, self.d_model)
        
        # 6. Output projektálás
        output = self.out_proj(attn_output)
        
        return output


class SimpleMLP(nn.Module):
    """Egyszerű FFN/MLP - STABIL"""
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = d_model * expansion
        
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Stabil inicializáció
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.02)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class StabilHybridLayer(nn.Module):
    """STABIL Hybrid Layer - Dinamikus Router-alapú SSM + Attention Fusion"""
    def __init__(self, config):
        super().__init__()
        
        # Norms
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Parallel Modules
        self.ssm = SimpleSSM(
            d_model=config.d_model,
            state_dim=config.ssm_state_dim,
            conv_kernel=config.ssm_conv_kernel,
            dropout=config.ssm_dropout
        )
        self.attn = SimpleFlashAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.attn_dropout
        )
        
        # Dinamikus Router
        self.router = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.SiLU(),
            nn.Linear(config.d_model // 4, 2)
        )
        
        # Erősebb szimmetria törés
        nn.init.normal_(self.router[2].weight, mean=0.0, std=0.1)
        # Az SSM irányba enyhe pozitív torzítás
        self.router[2].weight.data[0] += 0.5
        
        self.softmax = nn.Softmax(dim=-1)

        
        # MLP
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = SimpleMLP(
            d_model=config.d_model,
            expansion=config.ffn_expansion,
            dropout=config.ffn_dropout
        )
        
        self.residual_dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        residual = x
        x_norm = self.norm1(x)
        
        # Dinamikus Router logitok
        logits = self.router(x_norm)
        weights = self.softmax(logits)
        ssm_weight = weights[..., 0:1]
        attn_weight = weights[..., 1:2]
        
        # Router Kényszerítés (Load Balance): büntetjük, ha túl közel vannak 0.5-höz
        self.router_loss = torch.mean(torch.abs(ssm_weight - 0.5) + torch.abs(attn_weight - 0.5))
        
        ssm_out = self.ssm(x_norm)
        attn_out = self.attn(x_norm, mask=mask)
        
        x = x + self.residual_dropout(ssm_weight * ssm_out + attn_weight * attn_out)
        
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + self.residual_dropout(x)
        
        return x