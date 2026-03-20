"""
model.py
--------
Byte Riders | MAHE Mobility Challenge 2026
Track 1: Intent & Trajectory Prediction

Architecture:
    1. Temporal Encoder  – LSTM encodes each agent's (x,y) history
    2. Social Pooling    – Graph Attention aggregates neighbour context
    3. Multi-Modal Decoder – Generates K=3 future trajectory hypotheses
                            using a Conditional VAE (CVAE) head

Input shapes  (all batched):
    obs    : (B, OBS_LEN,  2)
    neigh  : (B, N_AGENTS, OBS_LEN, 2)

Output shapes:
    pred_trajs : (B, K, PRED_LEN, 2)   – K=3 candidate trajectories
    log_probs  : (B, K)                – log-probability of each mode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Constants (must match data_loader.py)
# ─────────────────────────────────────────────
OBS_LEN    = 4    # 2s × 2Hz
PRED_LEN   = 6    # 3s × 2Hz
MAX_AGENTS = 20
K          = 3    # number of predicted modes


# ─────────────────────────────────────────────
# 1. Temporal Encoder
# ─────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder that maps a trajectory sequence → hidden vector.

    Args:
        input_dim  : coordinate dimensions (default 2 for x,y)
        hidden_dim : LSTM hidden size
        num_layers : stacked LSTM layers
        dropout    : dropout between layers (0 if num_layers==1)
    """

    def __init__(self, input_dim=2, hidden_dim=64,
                 num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Linear(input_dim, hidden_dim)
        self.lstm  = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        """
        Args:
            x : (B, T, 2)
        Returns:
            h : (B, hidden_dim)   – final hidden state
        """
        x = F.relu(self.embed(x))          # (B, T, hidden_dim)
        _, (h, _) = self.lstm(x)           # h: (num_layers, B, hidden_dim)
        return h[-1]                        # (B, hidden_dim)


# ─────────────────────────────────────────────
# 2. Social Pooling (Graph Attention)
# ─────────────────────────────────────────────

class SocialAttention(nn.Module):
    """
    Single-head dot-product attention over neighbour encodings.
    The ego agent queries all neighbours and aggregates their context.

    Args:
        hidden_dim : dimension of agent encodings
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale  = hidden_dim ** 0.5

    def forward(self, ego_h, neigh_h, neigh_mask):
        """
        Args:
            ego_h      : (B, hidden_dim)
            neigh_h    : (B, N, hidden_dim)
            neigh_mask : (B, N) bool – True for valid neighbours
        Returns:
            ctx : (B, hidden_dim)   – aggregated social context
        """
        Q = self.q_proj(ego_h).unsqueeze(1)         # (B, 1, H)
        K = self.k_proj(neigh_h)                    # (B, N, H)
        V = self.v_proj(neigh_h)                    # (B, N, H)

        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, N)
        attn = attn.squeeze(1)                                 # (B, N)

        # Mask out padding neighbours with -inf before softmax
        attn = attn.masked_fill(~neigh_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        # If ALL neighbours are masked, softmax gives NaN → replace with 0
        attn = torch.nan_to_num(attn, nan=0.0)

        ctx = torch.bmm(attn.unsqueeze(1), V).squeeze(1)     # (B, H)
        return ctx


# ─────────────────────────────────────────────
# 3. Multi-Modal Decoder (CVAE)
# ─────────────────────────────────────────────

class MultiModalDecoder(nn.Module):
    """
    Generates K trajectory hypotheses from the fused context vector.

    Strategy:
        - A shared MLP maps context → K mode embeddings.
        - Each mode embedding is decoded by a GRU into PRED_LEN (x,y) steps.
        - A mode classifier assigns log-probabilities to each mode.

    Args:
        context_dim : input context size (ego + social)
        hidden_dim  : GRU hidden size
        pred_len    : number of future timesteps
        K           : number of predicted modes
    """

    def __init__(self, context_dim, hidden_dim=64,
                 pred_len=PRED_LEN, K=K):
        super().__init__()
        self.K        = K
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # Map fused context → K initial GRU hidden states
        self.mode_embed = nn.Linear(context_dim, K * hidden_dim)

        # Shared GRU decoder (processes one mode at a time)
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Output head: hidden → (x, y) per step
        self.out_proj = nn.Linear(hidden_dim, 2)

        # Mode probability head
        self.mode_clf = nn.Linear(context_dim, K)

    def forward(self, context):
        """
        Args:
            context : (B, context_dim)
        Returns:
            pred_trajs : (B, K, pred_len, 2)
            log_probs  : (B, K)
        """
        B = context.size(0)

        # (B, K * hidden_dim) → (B, K, hidden_dim)
        mode_h = self.mode_embed(context).view(B, self.K, self.hidden_dim)

        trajs = []
        for k in range(self.K):
            h0  = mode_h[:, k, :].unsqueeze(0)          # (1, B, hidden_dim)
            inp = torch.zeros(B, self.pred_len, 2,
                              device=context.device)     # start token = origin
            out, _ = self.gru(inp, h0)                  # (B, pred_len, hidden_dim)
            traj_k = self.out_proj(out)                  # (B, pred_len, 2)
            trajs.append(traj_k)

        pred_trajs = torch.stack(trajs, dim=1)           # (B, K, pred_len, 2)
        log_probs  = F.log_softmax(
            self.mode_clf(context), dim=-1
        )                                                # (B, K)

        return pred_trajs, log_probs


# ─────────────────────────────────────────────
# 4. Full Model
# ─────────────────────────────────────────────

class TrajectoryPredictor(nn.Module):
    """
    End-to-end trajectory prediction model for Byte Riders.

    Forward pass:
        obs    (B, OBS_LEN, 2)
        neigh  (B, MAX_AGENTS, OBS_LEN, 2)
        →
        pred_trajs (B, K, PRED_LEN, 2)
        log_probs  (B, K)

    Args:
        hidden_dim   : shared hidden size for all sub-modules
        num_layers   : LSTM layers in temporal encoder
        dropout      : dropout rate
        K            : number of trajectory modes
    """

    def __init__(self, hidden_dim=64, num_layers=1,
                 dropout=0.1, K=K):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Ego encoder
        self.ego_encoder   = TemporalEncoder(
            input_dim=2, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        )
        # Shared neighbour encoder (same weights for all neighbours)
        self.neigh_encoder = TemporalEncoder(
            input_dim=2, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
        )
        # Social attention
        self.social_attn   = SocialAttention(hidden_dim=hidden_dim)

        # Fusion: [ego_h | social_ctx] → fused_ctx
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Multi-modal decoder
        self.decoder = MultiModalDecoder(
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            pred_len=PRED_LEN,
            K=K,
        )

    def forward(self, obs, neigh):
        """
        Args:
            obs   : (B, OBS_LEN, 2)
            neigh : (B, N, OBS_LEN, 2)

        Returns:
            pred_trajs : (B, K, PRED_LEN, 2)
            log_probs  : (B, K)
        """
        B, N, T, _ = neigh.shape

        # ── Ego encoding ──────────────────────────────
        ego_h = self.ego_encoder(obs)                # (B, H)

        # ── Neighbour encoding ────────────────────────
        # Flatten (B, N, T, 2) → (B*N, T, 2), encode, reshape back
        neigh_flat = neigh.view(B * N, T, 2)
        neigh_h    = self.neigh_encoder(neigh_flat)  # (B*N, H)
        neigh_h    = neigh_h.view(B, N, -1)          # (B, N, H)

        # Build validity mask: neighbour is "present" if any position != 0
        neigh_mask = neigh.abs().sum(dim=(-1, -2)) > 0  # (B, N)

        # ── Social attention ──────────────────────────
        social_ctx = self.social_attn(
            ego_h, neigh_h, neigh_mask
        )                                            # (B, H)

        # ── Fusion ────────────────────────────────────
        fused = self.fusion(
            torch.cat([ego_h, social_ctx], dim=-1)
        )                                            # (B, H)

        # ── Multi-modal decoding ──────────────────────
        pred_trajs, log_probs = self.decoder(fused)

        return pred_trajs, log_probs


# ─────────────────────────────────────────────
# Quick sanity check (run this file directly)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    B   = 8
    obs   = torch.randn(B, OBS_LEN, 2)
    neigh = torch.randn(B, MAX_AGENTS, OBS_LEN, 2)

    model = TrajectoryPredictor(hidden_dim=64, num_layers=1, dropout=0.1, K=K)
    model.eval()

    with torch.no_grad():
        pred_trajs, log_probs = model(obs, neigh)

    print("\n── Model output shapes ───────────────────")
    print(f"  pred_trajs : {pred_trajs.shape}")   # (8, 3, 6, 2)
    print(f"  log_probs  : {log_probs.shape}")    # (8, 3)

    # Verify shapes
    assert pred_trajs.shape == (B, K, PRED_LEN, 2), \
        f"Expected ({B},{K},{PRED_LEN},2), got {pred_trajs.shape}"
    assert log_probs.shape == (B, K), \
        f"Expected ({B},{K}), got {log_probs.shape}"

    # Verify log_probs sum to ~1 in probability space
    probs_sum = log_probs.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones(B), atol=1e-5), \
        f"Mode probabilities do not sum to 1: {probs_sum}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")
    print("──────────────────────────────────────────")
    print("model.py ✓  All assertions passed.")
