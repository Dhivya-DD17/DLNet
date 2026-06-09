import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import ai_edge_torch
from ai_edge_litert.interpreter import Interpreter


# ===== Data for Dynamic LNN =====
class UniversalSOHDataset(Dataset):
    def __init__(self, sequences, scaler):
        self.sequences = sequences
        self.scaler = scaler
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x_seq, y_seq, battery_id = self.sequences[idx]
        x_scaled = self.scaler.transform(x_seq.reshape(-1, 1)).flatten()
        y_scaled = self.scaler.transform(y_seq.reshape(-1, 1)).flatten()
        return (torch.tensor(x_scaled, dtype=torch.float32), 
                torch.tensor(y_scaled, dtype=torch.float32),
                battery_id)
    

from torchdiffeq import odeint
# ===== Dynamic LNN model =====
class UniversalLNN_SOH(nn.Module):
    def __init__(self, input_len=100, hidden_dim=128, output_len=100):
        super().__init__()
        
        
        self.encoder = nn.Sequential(
            nn.Linear(input_len, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        
        self.dynamics = LiquidNeuron(hidden_dim)

        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, output_len)
        )
        
        
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_seq):
        batch_size = x_seq.shape[0]
        device = x_seq.device
        
        x_encoded = self.encoder(x_seq)
        
        self.dynamics.u = x_seq
        
        try:    
            self.dynamics.to(device)
            
            time_steps = 10
            t = torch.linspace(0, 1, time_steps).to(device)
            
            trajectory = odeint(self.dynamics, x_encoded, t, method='euler', rtol=1e-4, atol=1e-6)
            x_dynamic = trajectory[-1]  
            
        except Exception as e:
            # If ODE failed, use fallback
            print(f"⚠️ ODE failed, use fallback: {e}")
            with torch.no_grad():
                x_dynamic = torch.tanh(self.dynamics.W(x_encoded))
        
        # Residual connection
        x_combined = x_dynamic + self.residual_weight * x_encoded
        
        # Decoding
        output = self.decoder(x_combined)
        
        return output

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
    
class StudentModel(UniversalTransformer_SOH):
    def __init__(self, input_len=100, hidden_dim=64, output_len=100):
        super().__init__(input_len, hidden_dim, output_len)


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
                 num_levels=5, kernel_size=5, dropout=0.05, pool='attention'):
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

# ===== Data loading =====
def load_all_battery_data(data_dir, input_len=100, output_len=100, overlap_ratio=0.8):
    
    data_dir = Path(data_dir)
    file_paths = sorted(data_dir.glob("Bat*_SOH.xlsx"))
    
    all_sequences = []
    battery_info = {}
    
    # Calculate the step length
    step_size = max(1, int((input_len + output_len) * (1 - overlap_ratio)))
    
    print(f" Data loading:")
    print(f"  Input length: {input_len}, Output length: {output_len}")
    print(f"  Overlap ratio: {overlap_ratio*100:.1f}%, Step size: {step_size}")
    print(f"  Find {len(file_paths)} of battery files")
    
    for i, file_path in enumerate(file_paths):
        try:
            # Read data
            df = pd.read_excel(file_path)
            soh_data = df.iloc[:, 0].values
            
            # Cleaning data
            soh_data = soh_data[~np.isnan(soh_data)]  # Remove NaN
            if len(soh_data) < input_len + output_len:
                print(f"  ⚠️  {file_path.name}: The length is not enough: pass")
                continue
            
            # Data Smoothing
            if len(soh_data) > 5:
                from scipy.signal import savgol_filter
                soh_data = savgol_filter(soh_data, window_length=5, polyorder=2)
            
            battery_id = file_path.stem  
            sequence_count = 0
            
            # Generating training series
            for start_idx in range(0, len(soh_data) - input_len - output_len + 1, step_size):
                x_seq = soh_data[start_idx:start_idx + input_len]
                y_seq = soh_data[start_idx + input_len:start_idx + input_len + output_len]
                
                # Quality check
                if (x_seq.min() > 0.5 and x_seq.max() <= 1.1 and 
                    y_seq.min() > 0.5 and y_seq.max() <= 1.1):
                    all_sequences.append((x_seq, y_seq, battery_id))
                    sequence_count += 1
            
            battery_info[battery_id] = {
                'total_cycles': len(soh_data),
                'sequences': sequence_count,
                'soh_range': (soh_data.min(), soh_data.max())
            }
            
            print(f"  ✅ {battery_id}: {len(soh_data)} cycles → {sequence_count} sequences")
            
        except Exception as e:
            print(f"  ❌ Fail to deal with {file_path.name}: {e}")
    
    print(f"\n📈 Data summary:")
    print(f"  Total sereis number: {len(all_sequences)}")
    print(f"  Valid battery number: {len(battery_info)}")
    
    return all_sequences, battery_info

# ===== Train_test split =====
def smart_train_test_split(sequences, test_size=0.2, random_state=42):
    
    # Divide based on battery 
    battery_sequences = {}
    for seq in sequences:
        x, y, battery_id = seq
        if battery_id not in battery_sequences:
            battery_sequences[battery_id] = []
        battery_sequences[battery_id].append(seq)
    
    train_sequences = []
    test_sequences = []
    
    random.seed(random_state)
    
    for battery_id, battery_seqs in battery_sequences.items():
        n_test = max(1, int(len(battery_seqs) * test_size))
        test_indices = random.sample(range(len(battery_seqs)), n_test)
        
        for i, seq in enumerate(battery_seqs):
            if i in test_indices:
                test_sequences.append(seq)
            else:
                train_sequences.append(seq)
    
    print(f"📊 Split results:")
    print(f"  Train dataset: {len(train_sequences)} series")
    print(f"  Test dataset: {len(test_sequences)} series")
    
    return train_sequences, test_sequences


if __name__ == "__main__":
    # Parameters setting
    INPUT_LEN = 100
    OUTPUT_LEN = 100
    HIDDEN_DIM = 16
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 1e-3
    OVERLAP_RATIO = 0.8 
    BASE_DIR = Path(__file__).parent
  
    # Data direction
    data_dir = BASE_DIR/"smoothed_soh_excel"

    # model_path = BASE_DIR/f"models/{model_name}/{model_name}_Euler_student.pth"
    
    print("🔬 Universal LNN SOH Predictor")
    print(f"📋 Setting: {INPUT_LEN}→{OUTPUT_LEN}, Hidden layer:{HIDDEN_DIM}, Batch size:{BATCH_SIZE}")
    print(f"🎯 Object: Train the Dynamic LNN with MIT dataset\\n")
    
    # Load data
    print("📁 Loading battery data...")
    try:
        all_sequences, battery_info = load_all_battery_data(
            data_dir, INPUT_LEN, OUTPUT_LEN, OVERLAP_RATIO
        )
        
        if len(all_sequences) == 0:
            raise ValueError("Cannot find the series data")
            
    except Exception as e:
        print(f"❌ Failure on data loading: {e}")
        print("🔄 Generate simulated data (NOT recommended)...")
        
        # Generate simulated data
        all_sequences = []
        for i in range(20):  
            cycles = np.arange(0, 800)
            soh = 1.0 - 0.0002 * cycles - 0.0000005 * cycles**2
            soh += 0.02 * np.sin(cycles * 0.02) + 0.01 * np.random.randn(len(cycles))
            soh = np.clip(soh, 0.8, 1.0)
            
            for j in range(0, len(soh) - INPUT_LEN - OUTPUT_LEN + 1, 20):
                x_seq = soh[j:j+INPUT_LEN]
                y_seq = soh[j+INPUT_LEN:j+INPUT_LEN+OUTPUT_LEN]
                all_sequences.append((x_seq, y_seq, f"SimBat{i:03d}"))
    
    # Pre-process
    print("\\n🔧 Pre processing...")
    all_soh_data = []
    for x_seq, y_seq, _ in all_sequences:
        all_soh_data.extend(x_seq)
        all_soh_data.extend(y_seq)
    
    scaler = MinMaxScaler()
    scaler.fit(np.array(all_soh_data).reshape(-1, 1))
    print(f"📏 SoH data range: [{np.min(all_soh_data):.3f}, {np.max(all_soh_data):.3f}]")
    
    # Split data
    train_sequences, test_sequences = smart_train_test_split(all_sequences, test_size=0.2)
    
    train_dataset = UniversalSOHDataset(train_sequences, scaler)
    test_dataset = UniversalSOHDataset(test_sequences, scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    


    # --- Load your trained model class ---
    # from your_model_file import UniversalLNN_SOH
    input_len = 100
    hidden_dim = 128
    output_len = 100

    # Load PyTorch model
    # try: # Compressed model
    # MODELS = {"CNN", "TCN", "RNN", "LSTM", "GRU", "Transformer"}
    MODELS = {"LNN"}
    print("Float16----")
    for model_name in MODELS:
        model_path = BASE_DIR/f"models/{model_name}/{model_name}_optimal.pth"
        # if LNN: {model_name}_Euler_student.pth
        model=torch.load(model_path, map_location="cpu", weights_only = False)
        
        # except:# Teacher model
        #     model_class = globals()[f"Universal{model_name}_SOH"] 
        #     model = model_class(100, 128, 100)  
        #     model.load_state_dict(torch.load(model_path))
        model.eval()

        # # --- Sample input for tracing ---
        sample_inputs = (torch.randn(1, input_len),)  # batch=1 for Arduino

        def representative_dataset():
                    train_sequences, _ = smart_train_test_split(all_sequences, test_size=0.2)
                    train_dataset = UniversalSOHDataset(train_sequences, scaler)
                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
                    for batch_idx, (x_batch, y_batch, _) in enumerate(train_loader):
                        x_batch = x_batch.cpu().numpy()  # Move to CPU and convert to numpy
                        for i in range(x_batch.shape[0]): # Yield one sample at a time, shape (1, input_len)
                            yield [np.expand_dims(x_batch[i].astype(np.float32), axis=0)]

        # --- TFLite converter flags ---
        tfl_converter_flags = {
            'optimizations': [tf.lite.Optimize.DEFAULT],
            'representative_dataset': representative_dataset,
            'target_spec.supported_ops': [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
            'inference_input_type': tf.int8,
            'inference_output_type': tf.int8,
            "_experimental_disable_per_channel": True  # avoids some unsupported ops issues
        }

        # --- Convert to fully INT8 TFLite using ai_edge_torch ---

        tfl_fullint_model = ai_edge_torch.convert(
                model,
                sample_inputs,
                _ai_edge_converter_flags=tfl_converter_flags
        )

        os.makedirs("/app/models", exist_ok=True)
        tfl_fullint_model.export(f"/app/models/{model_name}_teacher.tflite")
        print(f"=============✅ TFLite INT8 model exported successfully===========")
            
        # interpreter = Interpreter(model_path=os.path.join(BASE_DIR, f"models/{model_name}_optimal1.tflite"))
        # interpreter.allocate_tensors()
        # for detail in interpreter.get_tensor_details():
        #     print(detail['name'], detail['dtype'])
        

        # # --- TFLite converter flags for float16 quantization ---
        # tfl_converter_flags = {
        #     'optimizations': [tf.lite.Optimize.DEFAULT],
        #     'target_spec.supported_types': [tf.float16],  # float16 instead of INT8
        #     'inference_input_type': tf.float32,
        #     'inference_output_type': tf.float32,
        #     "_experimental_disable_per_channel": True  # optional, avoids some unsupported ops issues
        # }

        # # --- Convert to float16 TFLite using ai_edge_torch ---
        # tfl_float16_model = ai_edge_torch.convert(
        #         model,
        #         sample_inputs,
        #         _ai_edge_converter_flags=tfl_converter_flags
        # )

        # os.makedirs("/app/models", exist_ok=True)
        # tfl_float16_model.export(f"/app/models/{model_name}_optimal_float16.tflite")
        # print(f"=============✅ TFLite FLOAT16 model exported successfully===========")
