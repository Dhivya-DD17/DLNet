import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torchdiffeq import odeint
from pathlib import Path
import warnings
import random, json
import time
import os
from codecarbon import EmissionsTracker
import psutil

warnings.filterwarnings("ignore")

print("Library Versions:")
import sys
print(f"Python: {sys.version}")
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"torch: {torch.__version__}")

# sklearn version (scaler is imported from it)
import sklearn
print(f"scikit-learn: {sklearn.__version__}")

# torchdiffeq version
import importlib.metadata
try:
    print(f"torchdiffeq: {importlib.metadata.version('torchdiffeq')}")
except importlib.metadata.PackageNotFoundError:
    print("torchdiffeq: not installed")

print(f"codecarbon: {importlib.metadata.version('codecarbon')}")
print(f"psutil: {psutil.__version__}")

print("\nPython version info:")
import sys
print(sys.version)


# Loading the model frameworks
class LiquidNeuron(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(dim) * 0.5)
        self.beta = nn.Parameter(torch.ones(dim) * 0.1)
        
    def forward(self, t, x):
        device = x.device
        if hasattr(self, 'u') and self.u is not None:
            u_input = self.u.to(device)
            u_mean = torch.mean(u_input, dim=1, keepdim=True).expand(-1, x.shape[1])
            return -self.alpha.to(device) * x + self.beta.to(device) * torch.tanh(self.W(x) + u_mean)
        else:
            return -self.alpha.to(device) * x + self.beta.to(device) * torch.tanh(self.W(x))

class UniversalLNN_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.dynamics = LiquidNeuron(hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, output_len)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_seq):
        device = x_seq.device
        x_encoded = self.encoder(x_seq)
        self.dynamics.u = x_seq
        try:
            self.dynamics.to(device)
            t = torch.linspace(0, 1, 10).to(device)
            trajectory = odeint(self.dynamics, x_encoded, t, method='euler')
            x_dynamic = trajectory[-1]
        except:
            x_dynamic = torch.tanh(self.dynamics.W(x_encoded))
        x_combined = x_dynamic + self.residual_weight * x_encoded
        return self.decoder(x_combined)
    
class StudentLNN(UniversalLNN_SOH):
    def __init__(self, input_len=100, hidden_dim=2, output_len=100):
        super().__init__(input_len, hidden_dim, output_len)

#============Student Model===============================

class LiquidNeuronCompressed(nn.Module):
    """
    Fast diagonal + low-rank residual ODE replacement.
    Approximates integration with one cheap learned Euler step.
    """
    def __init__(self, dim, rank=8, dt=0.1):
        super().__init__()
        self.dt = dt
        # diagonal weights (O(d))
        self.w_diag = nn.Parameter(torch.randn(dim) * 0.1)
        # low-rank residual (O(d*r))
        self.U = nn.Parameter(torch.randn(dim, rank) * 0.1)
        self.V = nn.Parameter(torch.randn(dim, rank) * 0.1)
        self.alpha = nn.Parameter(torch.ones(dim) * 0.5)
        self.beta  = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, x, u=None):
        # Low-rank linear part
        wx = x * self.w_diag            # diagonal term
        if self.U is not None:
            wx += (x @ self.V) @ self.U.T / self.V.size(1)
        if u is not None:
            u_mean = torch.mean(u, dim=1, keepdim=True)
            wx = wx + u_mean.expand_as(wx)

        dx = -self.alpha * x + self.beta * torch.tanh(wx)
        # one learned Euler step
        return x + self.dt * dx

class UniversalLNN_SOH_odefree(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100, rank=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.dynamics = LiquidNeuronCompressed(hidden_dim, rank=rank)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Linear(128, output_len)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_seq):
        x_encoded = self.encoder(x_seq)
        # Direct one-step flow instead of odeint
        x_dynamic = self.dynamics(x_encoded, u=x_seq)
        x_combined = x_dynamic + self.residual_weight * x_encoded
        return self.decoder(x_combined)
    
# ===== Student Model (Smaller) =====
class StudentLNN_ode(UniversalLNN_SOH_odefree):
    def __init__(self, input_len=100, hidden_dim=2, output_len=100):
        super().__init__(input_len, hidden_dim, output_len)
#==============================================================================

#-----------------------GRU Model----------------------------------------------
class UniversalGRU_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100, num_layers=2):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim

        # GRU input size = 1 (SOH scalar per time step)
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Decoder to map hidden state → output sequence
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_len)
        )

    def forward(self, x):
        # x: [batch, input_len]
        x = x.unsqueeze(-1)  # → [batch, input_len, 1]
        _, h_last = self.encoder(x)  # h_last: [num_layers, batch, hidden_dim]
        h_last = h_last[-1]          # Take last GRU layer → [batch, hidden_dim]
        out = self.decoder(h_last)   # → [batch, output_len]
        return out


# ===== Student Model (Smaller) =====
class StudentGRU(UniversalGRU_SOH):
    def __init__(self, input_len=100, hidden_dim=64, output_len=100):
        super().__init__(input_len, hidden_dim, output_len)


# =====================================Transformers============================================
class UniversalTransformer_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100,
                 nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, pool='mean'):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.pool = pool

        # Input embedding: scalar → hidden_dim
        self.embedding = nn.Linear(1, hidden_dim)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.zeros(1, input_len, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_len)
        )

    def forward(self, x):
        # x: [batch, input_len]
        x = x.unsqueeze(-1)           # → [batch, input_len, 1]
        x = self.embedding(x)         # → [batch, input_len, hidden_dim]
        x = x + self.pos_encoding     # add positional info

        # Transformer encoding
        x_encoded = self.transformer(x)  # → [batch, input_len, hidden_dim]

        # Pooling over the sequence
        if self.pool == 'mean':
            x_pooled = x_encoded.mean(dim=1)  # → [batch, hidden_dim]
        elif self.pool == 'max':
            x_pooled, _ = x_encoded.max(dim=1)
        else:
            # fallback to last token
            x_pooled = x_encoded[:, -1, :]

        # Decode to full output sequence
        out = self.decoder(x_pooled)  # → [batch, output_len]
        return out
    
# class StudentModel(UniversalTransformer_SOH):
#     def __init__(self, input_len=100, hidden_dim=64, output_len=100):
#         super().__init__(input_len, hidden_dim, output_len)

# --------------------------------RNN-------------------------------------
class UniversalRNN_SOH(nn.Module):
    def __init__(self, input_len, hidden_dim, output_len, num_layers=2, dropout=0.1):
        super().__init__()

        # Replace LSTM with vanilla RNN
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'    # default, but explicit
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)   # prunable
        self.fc2 = nn.Linear(hidden_dim, output_len)   # final output (ignore)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(-1)  # (B, T, 1)

        # RNN has only h0, no c0
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size, device=x.device)

        out, hn = self.rnn(x, h0)
        h = hn[-1]  # last layer hidden state

        h = self.dropout(torch.relu(self.fc1(h)))
        y = self.fc2(h)
        return y


# --------------------------------------LSTM--------------------------------------
class UniversalLSTM_SOH(nn.Module):
    def __init__(self, input_len, hidden_dim, output_len, num_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)   # prunable
        self.fc2 = nn.Linear(hidden_dim, output_len)   # final output (ignore)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)

        out, (hn, _) = self.lstm(x, (h0, c0))
        h = hn[-1]

        h = self.dropout(torch.relu(self.fc1(h)))
        y = self.fc2(h)
        return y

# --------------------------------------CNN---------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Basic Residual CNN Block -----
class ResidualConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, dropout=0.05):
        super().__init__()
        padding = (kernel_size - 1) // 2  # preserve sequence length

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Skip connection
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# ----- Full CNN Model -----
class UniversalCNN_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100,
                 num_blocks=2, kernel_size=5, dropout=0.05, pool='mean'):
        super().__init__()
        self.pool_type = pool

        # Input embedding: scalar → hidden_dim
        self.embedding = nn.Linear(1, hidden_dim)

        # Build CNN layers
        layers = []
        for i in range(num_blocks):
            in_ch = hidden_dim if i > 0 else hidden_dim
            layers.append(
                ResidualConv1D(in_ch, hidden_dim, kernel_size=kernel_size, dropout=dropout)
            )
        self.cnn = nn.Sequential(*layers)

        # Optional attention pooling
        self.pool = nn.Linear(hidden_dim, 1) if pool == 'attention' else None

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )

    def forward(self, x):
        # x: [B, L] → [B, L, 1]
        x = x.unsqueeze(-1)
        x = self.embedding(x)           # [B, L, hidden_dim]
        x = x.transpose(1, 2)           # [B, hidden_dim, L] for Conv1d

        x = self.cnn(x)                 # [B, hidden_dim, L]
        x = x.transpose(1, 2)           # [B, L, hidden_dim] for pooling

        # Pooling over temporal dimension
        if self.pool_type == 'mean':
            x = x.mean(dim=1)
        elif self.pool_type == 'max':
            x, _ = x.max(dim=1)
        elif self.pool_type == 'attention' and self.pool is not None:
            weights = torch.softmax(self.pool(x), dim=1)  # [B, L, 1]
            x = (x * weights).sum(dim=1)
        else:
            x = x[:, -1, :]

        out = self.decoder(x)
        return out


# --------------------------------------TCN----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Chomp1d to trim padding -----
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

# ----- Temporal Block with optional multi-scale convs -----
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.05):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # Multi-scale convolution: kernel_size, kernel_size-2, kernel_size-4 (if >1)
        convs = []
        for k in [kernel_size, max(kernel_size-2,1), max(kernel_size-4,1)]:
            convs.append(
                nn.Conv1d(in_ch, out_ch, k, stride=1, padding=(k-1)*dilation, dilation=dilation)
            )
        self.convs = nn.ModuleList(convs)
        self.chomp = nn.ModuleList([Chomp1d((k-1)*dilation) for k in [kernel_size, max(kernel_size-2,1), max(kernel_size-4,1)]])
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # Skip connection
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = 0
        for conv, chomp in zip(self.convs, self.chomp):
            temp = conv(x)
            temp = chomp(temp)
            out += temp
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# ----- Attention Pooling -----
class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, L, H]
        weights = torch.softmax(self.attn(x), dim=1)  # [B, L, 1]
        x_pooled = (x * weights).sum(dim=1)          # [B, H]
        return x_pooled

# ----- Full TCN -----
class UniversalTCN_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100,
                 num_levels=2, kernel_size=5, dropout=0.05, pool='attention'):
        super().__init__()
        self.pool_type = pool

        # Input embedding: scalar → hidden_dim
        self.embedding = nn.Linear(1, hidden_dim)

        # Build TCN layers with increasing dilation
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = hidden_dim if i > 0 else hidden_dim
            layers.append(
                TemporalBlock(in_ch, hidden_dim, kernel_size, dilation, dropout)
            )
        self.tcn = nn.Sequential(*layers)

        # Pooling
        if pool == 'attention':
            self.pool = TemporalAttentionPool(hidden_dim)
        else:
            self.pool = None

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )

    def forward(self, x):
        # x: [B, L] → [B, L, 1]
        x = x.unsqueeze(-1)
        x = self.embedding(x)           # [B, L, hidden_dim]
        x = x.transpose(1, 2)           # [B, hidden_dim, L] for Conv1d

        x = self.tcn(x)                 # [B, hidden_dim, L]
        x = x.transpose(1, 2)           # [B, L, hidden_dim] for pooling

        # Pooling over temporal dimension
        if self.pool_type == 'mean':
            x = x.mean(dim=1)
        elif self.pool_type == 'max':
            x, _ = x.max(dim=1)
        elif self.pool_type == 'attention' and self.pool is not None:
            x = self.pool(x)
        else:
            x = x[:, -1, :]

        out = self.decoder(x)
        return out


# =======================================================================================
def load_model_and_scaler(model_path, data_dir):

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # model = UniversalGRU_SOH(100, HIDDEN_DIM, 100)
    # model = UniversalTransformer_SOH(100, HIDDEN_DIM, 100)
    model_class = globals()[f"Universal{model_name}_SOH"] 
    model = model_class(100, 128, 100)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    files = list(Path(data_dir).glob("Bat*_SOH.xlsx"))[:3]
    all_soh = []
    for f in files:
        try:
            soh = pd.read_excel(f).iloc[:, 0].values
            all_soh.extend(soh[~np.isnan(soh)][::20])  
        except: continue
    
    scaler = MinMaxScaler()
    scaler.fit(np.array(all_soh).reshape(-1, 1))
    
    return model, scaler, device

def predict_segment(model, scaler, device, input_soh, target_soh, name, mc_runs=50):

    # scale input
    input_scaled = scaler.transform(input_soh.reshape(-1, 1)).flatten()
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    #---------------------Original Code (without MC uncertainty)------------------------
    # with torch.no_grad():
    #     output_scaled = model(input_tensor).cpu().numpy().flatten()
    #     predicted_soh = scaler.inverse_transform(output_scaled.reshape(-1, 1)).flatten()
    
    # # Average Error for all 100 predicted SoH
    # mae = np.mean(np.abs(predicted_soh - target_soh))
    # rmse = np.sqrt(np.mean((predicted_soh - target_soh)**2))
    # mape = np.mean(np.abs((predicted_soh - target_soh) / target_soh)) * 100

    # # Error at 10th future SoH
    # mae_10 = np.mean(np.abs(predicted_soh[9] - target_soh[9]))
    # rmse_10 = np.sqrt(np.mean((predicted_soh[9] - target_soh[9])**2))
    # mape_10 = np.mean(np.abs((predicted_soh[9] - target_soh[9]) / target_soh[9])) * 100

    # # Error at 10th future SoH
    # mae_100 = np.mean(np.abs(predicted_soh[99] - target_soh[99]))
    # rmse_100 = np.sqrt(np.mean((predicted_soh[99] - target_soh[99])**2))
    # mape_100 = np.mean(np.abs((predicted_soh[99] - target_soh[99]) / target_soh[99])) * 100

    #-----------------------------------------------------------------------------------

    # --- Monte Carlo dropout runs ---
    mc_outputs = []
    # Setup emissions tracker if available
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error",output_dir = BASE_DIR /"results_unc1", output_file=f"emissions_{name}.csv", allow_multiple_runs=True)
       
    # record memory before/after (best-effort)
    mem_before = None
    mem_after = None
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024**2  # MB

    # Enable dropout during MC sampling by setting model.train()
    model.train()
    # But avoid grad computation
    with torch.no_grad():
        for _ in range(mc_runs):
            out_scaled = model(input_tensor).cpu().numpy().flatten()
            out_inv = scaler.inverse_transform(out_scaled.reshape(-1, 1)).flatten()
            mc_outputs.append(out_inv)
    # Restore eval mode
    model.eval()
    # memory after
    proc = psutil.Process(os.getpid())
    mem_after = proc.memory_info().rss / 1024**2  # MB

    # convert mc outputs
    mc_array = np.stack(mc_outputs, axis=0)  # (mc_runs, output_len)
    predicted_mean = np.mean(mc_array, axis=0)
    predicted_std = np.std(mc_array, axis=0)
    unc = np.mean(predicted_std)

    # # compute point-wise metrics between mean prediction and target
    mae = np.mean(np.abs(predicted_mean - target_soh))
    rmse = np.sqrt(np.mean((predicted_mean - target_soh)**2))
    mape = np.mean(np.abs((predicted_mean - target_soh) / target_soh)) * 100

    # Error at 10th future SoH
    mae_10 = np.mean(np.abs(predicted_mean[9] - target_soh[9]))
    mape_10 = np.mean(np.abs((predicted_mean[9] - target_soh[9]) / target_soh[9])) * 100

    # Error at 10th future SoH
    mae_100 = np.mean(np.abs(predicted_mean[99] - target_soh[99]))
    mape_100 = np.mean(np.abs((predicted_mean[99] - target_soh[99]) / target_soh[99])) * 100

    #=======================================================================
    # timing: keep the same logic, but guard cuda call if not available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Warm-up (important for CPU/GPU to stabilize caches)
    for _ in range(10):
        _ = model(input_tensor)
    # Timing
    tracker.start()
    start_time = time.perf_counter()
    with torch.no_grad():
        sam_scaled = model(input_tensor).cpu().numpy().flatten()
        sam_inv = scaler.inverse_transform(sam_scaled.reshape(-1, 1)).flatten()
    end_time = time.perf_counter()
    tracker.stop()
    inf_time = ((end_time - start_time) * 1000) # in ms
    #=======================================================================

    # Confidence: percent of true target points within 95% predictive interval
    ci_low = predicted_mean - 1.96 * predicted_std
    ci_high = predicted_mean + 1.96 * predicted_std
    within = np.logical_and(target_soh >= ci_low, target_soh <= ci_high)
    confidence_pct = 100.0 * np.sum(within) / len(target_soh)

    # Memory footprint: report end memory if available, else None
    mem_mb = None
    if (mem_before is not None) and (mem_after is not None):
        mem_mb = mem_after  # current RSS in MB
    elif mem_after is not None:
        mem_mb = mem_after

    return mc_array, predicted_mean, predicted_std, unc, mae, rmse, mape, inf_time, confidence_pct, mem_mb, mae_10, mape_10, mae_100, mape_100


def create_test_segments(data_dir, num_segments=12, seed=42, save_path="selected_segments.json"):

    # directly load the test segments from JSON for consistency across different platforms
 
    save_path = Path(BASE_DIR) / save_path
    if save_path.exists():
        print(f"📂 Found existing test segments at {save_path}. Reusing them for consistency.")
        with open(save_path, "r") as f:
            return json.load(f)
    else:
        print("JSON doesn't exists")
  

def test_segments(model_path, data_dir, name, num_segments=12):
    """Test multiple battery segments"""
    # Load model
    try:
        model, scaler, device = load_model_and_scaler(model_path, data_dir)     
    except:
        model=torch.load(model_path, map_location="cpu", weights_only = False)      
    #==================================================================
    # model=torch.load(model_path, map_location="cuda", weights_only = False)
    # model=torch.load(model_path, map_location="cpu", weights_only = False)
    #==================================================================
    # model = StudentLNN(INPUT_LEN, 64, OUTPUT_LEN)
    # model.load_state_dict(torch.load(model_path))
    files = list(Path(data_dir).glob("Bat*_SOH.xlsx"))[:3]
    print("Files found:", files)
    files = [str(DATA_DIR / f) for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")][:3]
# print("Files found:", files)

    all_soh = []
    for f in files:
        try:
            soh = pd.read_excel(f).iloc[:, 0].values
            all_soh.extend(soh[~np.isnan(soh)][::20])  # 每20个取1个
        except: continue
    
    scaler = MinMaxScaler()
    scaler.fit(np.array(all_soh).reshape(-1, 1))
    #==================================================================

    # If load_model_and_scaler was used previously, device would exist; ensure we set device:
    device = torch.device('cpu')

    print(f"✅ Model loaded successfully, device: {device}")

    # Create test segments
    test_segments = create_test_segments(data_dir, num_segments)
    print(f"📊 Created {len(test_segments)} test segments")

    results = []
    n_segments = len(test_segments)  # should be 16
    cols = 4
    rows = int(np.ceil(n_segments / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4), constrained_layout=True)
    axes = axes.flatten()
    axes = axes.flatten() 
    all_mc_outputs = []
            
    for i, segment in enumerate(test_segments):
        try:
            # Read battery data
            file_path = Path(data_dir) / f"{segment['battery']}.xlsx"
            soh_data = pd.read_excel(file_path).iloc[:, 0].values
            soh_data = soh_data[~np.isnan(soh_data)]

            # Extract test segment
            start_idx = segment['start']
            end_idx = min(segment['end'], len(soh_data))

            if end_idx - start_idx < 200:
                continue

            # Define input and target
            input_soh = soh_data[start_idx:start_idx + 100]
            target_soh = soh_data[start_idx + 100:start_idx + 200]

            # Predict with MC dropout and resource tracking
            (mc_array, pred_mean, pred_std, unc, mae, rmse, mape, inf_time,
             confidence_pct, mem_mb, mae_10, mape_10, mae_100, mape_100) = predict_segment(
                 model, scaler, device, input_soh, target_soh, name, mc_runs=100
             )
            all_mc_outputs.append(mc_array)

            results.append({
                'battery': segment['battery'],
                'start_cycle': start_idx,
                'soh_range': segment['soh_range'],
                'avg_soh': segment['avg_soh'],
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'uncertainty': unc,
                'confidence_pct': confidence_pct,
                'inf_time': inf_time,
                'mem_mb': mem_mb,
                'mae_10': mae_10,
                'mape_10': mape_10,
                'mae_100': mae_100,
                'mape_100': mape_100

            })
    
            ax = axes[i]
            cycles = np.arange(200)
            true_full = np.concatenate([input_soh, target_soh])
            pred_full = np.concatenate([input_soh, pred_mean])  # show mean for full

            ax.plot(cycles, true_full, 'k-', label='True SOH', linewidth=2.5, alpha=0.8)
            ax.plot(cycles[100:], pred_mean, 'r--', label='Predicted SOH (mean)', linewidth=2.5, alpha=0.9)

            # 95% CI shading
            ci_low = pred_mean - 1.96 * pred_std
            ci_high = pred_mean + 1.96 * pred_std
            ax.fill_between(cycles[100:], ci_low, ci_high, alpha=0.2, label='95% CI')

            ax.axvline(100, color='blue', linestyle=':', alpha=0.7, linewidth=2, label='Prediction Start')

            # Highlight prediction region
            ax.axvspan(100, 200, alpha=0.1, color='red', label='Prediction Zone')

            ax.set_title(f'{segment["battery"]} - {segment["soh_range"].upper()} SOH Range\n',
                      fontsize=12, pad=12)

            ax.set_xlabel('Relative Cycle', fontsize=12)
            ax.set_ylabel('SOH', fontsize=12)
            # plt.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Set the Y range
            y_min = min(np.min(true_full), np.min(pred_full)) - 0.01
            y_max = max(np.max(true_full), np.max(pred_full)) + 0.01
            ax.set_ylim(y_min, y_max)

        except Exception as e:
            print(f"❌ {segment['battery']}: {e}")
            continue

    all_mc_outputs = np.stack(all_mc_outputs, axis=0)
    q75 = np.percentile(all_mc_outputs, 75, axis=1)  # (num_batteries, 100)
    q25 = np.percentile(all_mc_outputs, 25, axis=1)
    iqr = np.mean(q75 - q25, axis=0)  # average IQR across batteries → shape: (100,)

    # Save all MC outputs for later plotting
    np.save(BASE_DIR / f"results_unc1/all_mc_{model_name_lower}T.npy", all_mc_outputs)

    # Save computed overall IQR
    np.save(BASE_DIR / f"results_unc1/overall_iqr_{model_name_lower}T.npy", iqr)
    # # Remove any unused axes if segments < rows*cols
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    # Show once at the end
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"results_unc1/plot{name}.png")
    plt.close()
    plt.show()
        
    
    # Print the statistical results
    if results:
        print(f"\n📈 Test results summary ({len(results)} segments):")
        summary_data = []
        main_summary = []
        
        # 总体统计 (overall stats)
        avg_mae = np.mean([r['mae'] for r in results])
        avg_rmse = np.mean([r['rmse'] for r in results])
        avg_mape = np.mean([r['mape'] for r in results])
        avg_inf = np.mean([r['inf_time'] for r in results])
        avg_conf = np.mean([r['confidence_pct'] for r in results])
        avg_mem = np.mean([r['mem_mb'] for r in results])
        avg_unc = np.mean([r['uncertainty'] for r in results])

        # new: 10 & 100 averages
        avg_mae_10 = np.mean([r['mae_10'] for r in results])
        avg_mape_10 = np.mean([r['mape_10'] for r in results])

        avg_mae_100 = np.mean([r['mae_100'] for r in results])
        avg_mape_100 = np.mean([r['mape_100'] for r in results])

        print(f"   Overall average: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.1f}%, "
            f"MAE_10={avg_mae_10:.4f}, MAPE_10={avg_mape_10:.1f}%, "
            f"MAE_100={avg_mae_100:.4f}, MAPE_100={avg_mape_100:.1f}%, "
            f"Avg Inf Time={avg_inf:.4f}ms, Avg Uncertainty={avg_unc:.4f}, Avg Conf(95%)={avg_conf:.2f}%, Mem (MB)={avg_mem:.4f}")
                
        # Read CSV file
        df_e = pd.read_csv(BASE_DIR / f"results_unc1/emissions_{name}.csv")

        summary_data.append({
            # 'Segment': ALPHA,
            # 'Sparsity': n,
            # 'Pruning Type': p,
            'Hidden_dim': STU_DIM,
            'Alpha': ALPHA,
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'MAPE (%)': avg_mape,
            'MAE_10': avg_mae_10,
            'MAPE_10 (%)': avg_mape_10,
            'MAE_100': avg_mae_100,
            'MAPE_100 (%)': avg_mape_100,
            'Avg Uncertainty' : avg_unc,
            'Avg Conf (%)': avg_conf,
            'Avg Inf Time (ms)': avg_inf,
            'Mem (MB)': avg_mem,
            'Model Size (kB)': os.path.getsize(model_path) / 1024,
            'Energy': df_e["energy_consumed"].mean(),
            'Emission': df_e["emissions"].mean(),
            'Count': len(results)
        })

        main_summary.extend(summary_data)
        file_path = BASE_DIR/ f"results_unc1/summary{name}.csv"

        # Create DataFrame
        df = pd.DataFrame(main_summary)
        # Append if file exists, otherwise write with header
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

        #         # # Based on SoH segments
        # soh_ranges = ['high', 'medium', 'low', 'very_low']
        # for soh_range in soh_ranges:
        #     range_results = [r for r in results if r['soh_range'] == soh_range]
        #     if range_results:
        #         range_mae = np.mean([r['mae'] for r in range_results])
        #         range_rmse = np.mean([r['rmse'] for r in range_results])
        #         range_mape = np.mean([r['mape'] for r in range_results])
        #         range_mae_10 = np.mean([r['mae_10'] for r in range_results])
        #         range_mape_10 = np.mean([r['mape_10'] for r in range_results])
        #         range_mae_100 = np.mean([r['mae_100'] for r in range_results])
        #         range_mape_100 = np.mean([r['mape_100'] for r in range_results])
        #         avg_soh = np.mean([r['avg_soh'] for r in range_results])
        #         avg_conf_range = np.mean([r['confidence_pct'] for r in range_results])
        #         avg_unc_range = np.mean([r['uncertainty'] for r in range_results])

        #         print(f"   {soh_range.upper()} SOH interval: "
        #             f"MAE={range_mae:.4f}, RMSE={range_rmse:.4f}, MAPE={range_mape:.4f}  "
        #             f"MAE_10={range_mae_10:.4f}, MAPE_10={range_mape_10:.4f}, "
        #             f"MAE_100={range_mae_100:.4f}, MAPE_100={range_mape_100:.4f} "
        #             f"(Average SOH:{avg_soh:.3f}, {len(range_results)} segments, Avg Uncertainty={avg_unc_range:.4f}, Avg Conf: {avg_conf_range:.2f}%)")

        #         summary_data.append({
        #             'Segment': f"{soh_range.upper()} SOH",
        #             'MAE': range_mae,
        #             'RMSE': range_rmse,
        #             'MAPE (%)': range_mape,
        #             'MAE_10': range_mae_10,
        #             'MAPE_10 (%)': range_mape_10,
        #             'MAE_100': range_mae_100,
        #             'MAPE_100 (%)': range_mape_100,
        #             'Avg Inf Time (ms)': np.nan,
        #             'Avg Uncertainty' : avg_unc_range,
        #             'Avg Conf (%)': avg_conf_range,
        #             'Mem (MB)': np.nan,
        #             'Count': len(range_results),
        #             'Average SOH': avg_soh
        #         })
 
        # # Best and worst
        # best = min(results, key=lambda x: x['mae'])
        # worst = max(results, key=lambda x: x['mae'])
        # print(f"   Bast segments: {best['battery']} ({best['soh_range']}, SOH:{best['avg_soh']:.3f}, cycle {best['start_cycle']}) MAE={best['mae']:.4f}")
        # print(f"   Worst segments: {worst['battery']} ({worst['soh_range']}, SOH:{worst['avg_soh']:.3f}, cycle {worst['start_cycle']}) MAE={worst['mae']:.4f}")

        # summary_data.append({
        #     'Segment': 'Best Segment',
        #     'Battery': best['battery'],
        #     'SOH Range': best['soh_range'],
        #     'SOH': best['avg_soh'],
        #     'Cycle': best['start_cycle'],
        #     'MAE': best['mae']
        # })
        # summary_data.append({
        #     'Segment': 'Worst Segment',
        #     'Battery': worst['battery'],
        #     'SOH Range': worst['soh_range'],
        #     'SOH': worst['avg_soh'],
        #     'Cycle': worst['start_cycle'],
        #     'MAE': worst['mae']
        # })


        # # Save to CSV
        # df = pd.DataFrame(summary_data)
        # df.to_csv(f"results_unc/summary_ori{name}.csv", index=False)
        # print("Saved summary to summary.csv")

    return results

# Main
if __name__ == "__main__":
    INPUT_LEN = 100
    OUTPUT_LEN = 100
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    OVERLAP_RATIO = 0.8
    STU_DIM = 64
    ALPHA= 0.8
    EPOCH = 200
    BASE_DIR = Path(__file__).parent


    model_name = "CNN"
    model_name_lower = model_name.lower()
    try:
        model_class = globals()[f"Universal{model_name}_SOH"]
        StudentModel = type(
            "StudentModel",   # class name
            (model_class,),    # tuple of base classes
            {
                "__init__": lambda self, input_len=100, hidden_dim=64, output_len=100: 
                    model_class.__init__(self, input_len, hidden_dim, output_len)
            }
        )
    except:
        print("Other models")

    STU_HIDDEN_DIM = {f"{model_name}"}
    ALPHA_VAL = {"Ori"}
    # STU_HIDDEN_DIM= {"MSE", "Cos"}
    # ALPHA_VAL = {64, 32, 16, 8, 4, 2}
    # STU_HIDDEN_DIM= {"OdeC64"}
    # ALPHA_VAL = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
   
    
    EPOCH = 200
    for STU_DIM in STU_HIDDEN_DIM:
        for ALPHA in ALPHA_VAL:
            try:
                # MODEL_PATH = BASE_DIR/f"models/{model_name}/KDPru/MSE/full_stu_hd16.pth"
                
                MODEL_PATH = BASE_DIR/f"models/{model_name}/best_universal_{model_name}_soh.pth"
                # MODEL_PATH = BASE_DIR/f"models/LNN/LNN_Euler_student.pth"
                # MODEL_PATH = BASE_DIR/f"models/{model_name}/KD/{STU_DIM}/full_stu_hd{ALPHA}.pth"
                DATA_DIR = BASE_DIR/"smoothed_soh_excel"
                files = [str(DATA_DIR / f) for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")][:3]
                print("Files found:", files)

                random.seed(42)
                # name = f"ad_hd{STU_DIM}_a{ALPHA}"
                name = f"{ALPHA}_s{STU_DIM}"
                print(f"---------------------------------{name}----------------------------------")
                print("🧪 LNN SOH Segment Tester")
                print("Testing segments from different batteries and SoH ranges (100 → 100 cycles)")

                results = test_segments(MODEL_PATH, DATA_DIR, name, num_segments=16)  # Increase number of test segments to cover more SoH ranges
                print("\n✅ Segment testing completed!")
                            
            except Exception as e:
                print(f"\n❌ Failed: {e}")

