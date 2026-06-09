import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf        
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings
import random
import time
import os
from codecarbon import EmissionsTracker
import psutil
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from ai_edge_litert.interpreter import Interpreter


#============Student Model===============================

class LiquidNeuronCompressed(nn.Module):
    def __init__(self, dim, rank=8, dt=0.05):
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
    def __init__(self, input_len=100, hidden_dim=128, output_len=100, rank=None):
        super().__init__()
        if rank==None: rank=hidden_dim
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

def load_tflite16_model_and_scaler(model_path, data_dir):
    """Load TFLite interpreter and fit a MinMaxScaler (unchanged behavior but returns interpreter + details)."""
    # interpreter = tf.lite.Interpreter(model_path=str(model_path))
    # interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "model", "model.tflite"))
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("INPUT DETAILS:\n", input_details)
    print("OUTPUT DETAILS:\n", output_details)

    # Fit scaler on some representative SOH samples (same logic as before)
    files = list(Path(data_dir).glob("Bat*_SOH.xlsx"))[:3]
    all_soh = []
    for f in files:
        try:
            soh = pd.read_excel(f).iloc[:, 0].values
            soh = soh[~np.isnan(soh)]
            # sample down to avoid huge arrays but keep distribution
            if len(soh) > 0:
                all_soh.extend(soh[::max(1, len(soh)//1000)])
        except:
            continue

    if len(all_soh) == 0:
        # fallback so scaler doesn't fail
        all_soh = [0.9, 1.0, 0.85, 0.8]

    scaler = MinMaxScaler()
    scaler.fit(np.array(all_soh).reshape(-1, 1))
        # Min and Max used for scaling
    data_min = scaler.data_min_[0]
    data_max = scaler.data_max_[0]

    print("Min value:", data_min)
    print("Max value:", data_max)


    return interpreter, input_details, output_details, scaler



def predict_segment_tflite16(interpreter, input_details, output_details, scaler,
                           input_soh, target_soh, name, mc_runs=1):
    """
    Run inference with a TFLite interpreter while handling quantized input/output properly.
    Returns: predicted_mean, predicted_std, unc, mae, rmse, mape, inf_time, confidence_pct, mem_mb, mae_10, mape_10, mae_100, mape_100
    """
    # Scale input (0-1) using your fitted scaler
    input_scaled = scaler.transform(input_soh.reshape(-1, 1)).flatten().astype(np.float32)

    # Build a tensor shaped exactly like the interpreter expects
    expected_shape = input_details[0]['shape']  # e.g. [1,100] or [1,100,1]
    # Some interpreters use -1 for batch dim; replace -1 with 1 if present
    shape = [1 if (int(s) == -1) else int(s) for s in expected_shape]
    # Flatten scaled data then reshape to expected shape (preserve last dims)
    flat = input_scaled.flatten()
    try:
        input_tensor = np.array(flat).reshape(shape).astype(np.float32)
    except Exception:
        # fallback: try (1, length)
        input_tensor = np.array(flat).reshape((1, -1)).astype(np.float32)

    # Inspect quantization params and dtype for input
    in_dtype = input_details[0]['dtype']
    in_quant = input_details[0].get('quantization', (0.0, 0))
    in_scale, in_zero_point = (float(in_quant[0]), int(in_quant[1])) if in_quant is not None else (0.0, 0)

    out_dtype = output_details[0]['dtype']
    out_quant = output_details[0].get('quantization', (0.0, 0))
    out_scale, out_zero_point = (float(out_quant[0]), int(out_quant[1])) if out_quant is not None else (0.0, 0)

    # Utility: quantize float -> model dtype
    def quantize_input(x_float, scale, zero_point, dtype):
        if scale == 0.0:
            # model probably expects float32, no quantization
            return x_float.astype(dtype)
        x_q = np.round(x_float / scale + zero_point)
        if np.issubdtype(dtype, np.signedinteger):
            # clip to int8 range
            info = np.iinfo(dtype)
            x_q = np.clip(x_q, info.min, info.max)
        else:
            info = np.iinfo(dtype)
            x_q = np.clip(x_q, info.min, info.max)
        
        return x_q.astype(dtype)

    # Utility: dequantize output -> float
    def dequantize_output(y_q, scale, zero_point):
        if scale == 0.0:
            return y_q.astype(np.float32)
        return (y_q.astype(np.float32) - zero_point) * scale

    # Prepare MC outputs
    mc_outputs = []
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024**2

    # Pre-quantize the input if required
    if in_scale != 0.0 and (in_dtype in [np.int8, np.uint8, np.int16, np.uint16]):
        q_input = quantize_input(input_tensor.astype(np.float32), in_scale, in_zero_point, in_dtype)
        # print(f"Input:{input_tensor}")
        # print(f"Input Quantized:{q_input}")
        # Create a DataFrame
        input_flat = np.array(input_tensor).reshape(-1)
        q_input_flat = np.array(q_input).reshape(-1)


    else:
        # interpreter may support float input
        print("------Float16----------------")
        q_input = input_tensor.astype(in_dtype)

    # Run repeated inferences (deterministic) — we still run mc_runs for your uncertainty proxy
    for _ in range(mc_runs):
        try:
            # Ensure tensor shape matches exactly what the model expects
            # Some TFLite builds require resizing when first run; we assume allocate_tensors() already done.
            interpreter.set_tensor(input_details[0]['index'], q_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
        except Exception as e:
            # If interpreter complains about shapes / dtype, try converting to expected shape exactly
            try:
                correct_shape_input = np.array(flat).reshape(input_details[0]['shape']).astype(in_dtype)
                interpreter.set_tensor(input_details[0]['index'], correct_shape_input)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

            except Exception as e2:
                raise RuntimeError(f"TFLite invocation failed: {e} / fallback: {e2}")

        # output_data might be shape (1,100) or (1,100,1) etc. Flatten to 1D
        output_flat = np.array(output_data).reshape(-1)

        # Dequantize if needed
        if out_scale != 0.0 and (out_dtype in [np.int8, np.uint8, np.int16, np.uint16]):
            out_inv = dequantize_output(output_flat, out_scale, out_zero_point)
            out_quant = np.array(output_flat).reshape(-1)
            out_dequant = np.array(out_inv).reshape(-1)
            # print(f"Output Quant: {output_flat}")
            # print(f"Output final: {out_inv}")
        else:
            print("------Float16----------------")
            out_inv = output_flat.astype(np.float32)

        # If the model outputs normalized values, inverse transform them using scaler
        # We expect the output to be same scaling as scaler (single-channel)
        try:
            out_inv = scaler.inverse_transform(out_inv.reshape(-1, 1)).flatten()
        except Exception:
            # If inverse_transform fails (shape mismatch), try reshape to (-1,1)
            out_inv = scaler.inverse_transform(np.array(out_inv).reshape(-1, 1)).flatten()
        # print(f"Output final: {out_inv}")
        out_scaled = np.array(out_inv).reshape(-1)
        # df = pd.DataFrame({
        #     'Input_Float': input_flat,
        #     'Input_Quantized': q_input_flat,
        #     'Output_Quant': out_quant,
        #     'Output_DeQuant': out_dequant,
        #     'Output_Scaled': out_scaled,
        # })
        # Save to CSV
        # df.to_csv('PythonInOut.csv', index=False)
        # print("Saved Input and Quantized Input to input_quantized.csv")

        mc_outputs.append(out_inv)

    mem_after = proc.memory_info().rss / 1024**2

    # Statistics across MC runs
    mc_array = np.stack(mc_outputs, axis=0)  # shape: (mc_runs, output_len)
    predicted_mean = np.mean(mc_array, axis=0)
    predicted_std  = np.std(mc_array, axis=0)
    unc = np.mean(predicted_std)

    mae = np.mean(np.abs(predicted_mean - target_soh))
    rmse = np.sqrt(np.mean((predicted_mean - target_soh) ** 2))
    mape = np.mean(np.abs((predicted_mean - target_soh) / target_soh)) * 100

    # compute specific-cycle metrics if lengths permit (guard indices)
    def safe_idx(arr, idx):
        return float(arr[idx]) if len(arr) > idx else np.nan

    mae_10 = abs(safe_idx(predicted_mean, 9) - safe_idx(target_soh, 9)) if len(target_soh) > 9 else np.nan
    mape_10 = (abs(safe_idx(predicted_mean, 9) - safe_idx(target_soh, 9)) / safe_idx(target_soh, 9) * 100) if len(target_soh) > 9 and safe_idx(target_soh, 9) != 0 else np.nan
    mae_100 = abs(safe_idx(predicted_mean, 99) - safe_idx(target_soh, 99)) if len(target_soh) > 99 else np.nan
    mape_100 = (abs(safe_idx(predicted_mean, 99) - safe_idx(target_soh, 99)) / safe_idx(target_soh, 99) * 100) if len(target_soh) > 99 and safe_idx(target_soh, 99) != 0 else np.nan

    # Measure one inference time and record emissions (your existing approach)
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error",
                               output_dir=BASE_DIR / "results_tflite16",
                               output_file=f"emissions_{name}.csv",
                               allow_multiple_runs=True)
    tracker.start()
    start_time = time.perf_counter()
    # run a single invocation (use q_input)
    interpreter.set_tensor(input_details[0]['index'], q_input)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.perf_counter()
    tracker.stop()

    inf_time = ((end_time - start_time) * 1000)  # ms
    ci_low = predicted_mean - 1.96 * predicted_std
    ci_high = predicted_mean + 1.96 * predicted_std
    within = np.logical_and(target_soh >= ci_low, target_soh <= ci_high)
    confidence_pct = 100.0 * np.sum(within) / len(target_soh)

    mem_mb = mem_after
    return predicted_mean, predicted_std, unc, mae, rmse, mape, inf_time, confidence_pct, mem_mb, mae_10, mape_10, mae_100, mape_100

import json

def create_test_segments(data_dir, num_segments=12, seed=42, save_path="selected_segments.json"):

    # directly load the test segments from JSON for consistency across different platforms
 
    save_path = Path(BASE_DIR) / save_path
    if save_path.exists():
        print(f"📂 Found existing test segments at {save_path}. Reusing them for consistency.")
        with open(save_path, "r") as f:
            return json.load(f)
    else:
        print("JSON doesn't exists")
  

def test_segments_tflite16(model_path, data_dir, name, num_segments=16):
    """Test multiple battery segments using TFLite"""
    interpreter, input_details, output_details, scaler = load_tflite16_model_and_scaler(model_path, data_dir)
    print(f"✅ TFLite model loaded successfully: {model_path.name}")

    test_segments = create_test_segments(data_dir, num_segments)
    print(f"📊 Created {len(test_segments)} test segments")

    results = []
    n_segments = len(test_segments)
    cols = 4
    rows = int(np.ceil(n_segments / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 4), constrained_layout=True)
    axes = axes.flatten()

    for i, segment in enumerate(test_segments):
        try:
            file_path = Path(data_dir) / f"{segment['battery']}.xlsx"
            soh_data = pd.read_excel(file_path).iloc[:, 0].values
            soh_data = soh_data[~np.isnan(soh_data)]

            start_idx = segment['start']
            end_idx = min(segment['end'], len(soh_data))
            if end_idx - start_idx < 200:
                continue

            input_soh = soh_data[start_idx:start_idx + 100]
            target_soh = soh_data[start_idx + 100:start_idx + 200]

            (pred_mean, pred_std, unc, mae, rmse, mape, inf_time,
             confidence_pct, mem_mb, mae_10, mape_10, mae_100, mape_100) = predict_segment_tflite16(
                 interpreter, input_details, output_details, scaler, input_soh, target_soh, name, mc_runs=1
             )

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
            pred_full = np.concatenate([input_soh, pred_mean])

            ax.plot(cycles, true_full, 'k-', label='True SOH', linewidth=2.5, alpha=0.8)
            ax.plot(cycles[100:], pred_mean, 'r--', label='Predicted SOH (mean)', linewidth=2.5, alpha=0.9)

            ci_low = pred_mean - 1.96 * pred_std
            ci_high = pred_mean + 1.96 * pred_std
            ax.fill_between(cycles[100:], ci_low, ci_high, alpha=0.2, label='95% CI')

            ax.axvline(100, color='blue', linestyle=':', alpha=0.7, linewidth=2)
            ax.axvspan(100, 200, alpha=0.1, color='red', label='Prediction Zone')

            ax.set_title(f'{segment["battery"]} - {segment["soh_range"].upper()} SOH Range\n', fontsize=12, pad=12)
            ax.set_xlabel('Relative Cycle', fontsize=12)
            ax.set_ylabel('SOH', fontsize=12)
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"❌ {segment['battery']}: {e}")
            continue

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"results_tflite16/plot{name}.png")
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
        df_e = pd.read_csv(BASE_DIR / f"results_tflite16/emissions_{name}.csv")

        summary_data.append({
            'Segment': ALPHA,
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
        file_path = BASE_DIR/ f"results_tflite16/summary{name}.csv"

        # Create DataFrame
        df = pd.DataFrame(main_summary)
        # Append if file exists, otherwise write with header
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

    return results

#=============================================
# Run Loop: same structure, only .tflite models
#=============================================
INPUT_LEN = 100
OUTPUT_LEN = 100
# BASE_DIR = Path(r"C:\Users\Desktop\Final_Project\EntroLNN_code\Model Compression")
BASE_DIR = Path(__file__).parent
# STU_HIDDEN_DIM = {0.1, 0.3, 0.8}
# ALPHA_VAL = {"Dis_OdeD4", "Cos_OdeC16"}
# STU_HIDDEN_DIM = {"rt6"}
# ALPHA_VAL = {"tran"}
# STU_HIDDEN_DIM = {"CNN", "TCN", "RNN", "LSTM", "GRU", "Transformer", "LNN"}
STU_HIDDEN_DIM = {"GRU"}
ALPHA_VAL = {"optimal_tflite"}

for STU_DIM in STU_HIDDEN_DIM:
    for ALPHA in ALPHA_VAL:
#         try:
#             MODEL_PATH = BASE_DIR / f"Tflite_conv/{ALPHA}_h{STU_DIM}.tflite"
#             DATA_DIR = BASE_DIR / "smoothed_soh_excel"
#             test_segments_tflite16(MODEL_PATH, DATA_DIR, f"{ALPHA}_hd{STU_DIM}")
#         except Exception as e:
#             print(f"❌ Failed for {ALPHA} hd{STU_DIM}: {e}")



        try:
                    # MODEL_PATH = BASE_DIR / "models/Transformer/transformer_optimal6.tflite"
                    MODEL_PATH = BASE_DIR/ f"models/{STU_DIM}/{STU_DIM}_optimal_float16.tflite"
                    DATA_DIR = BASE_DIR / "smoothed_soh_excel"
                    test_segments_tflite16(MODEL_PATH, DATA_DIR, f"{ALPHA}_hd{STU_DIM}")
        except Exception as e:
                    print(f"❌ Failed for {ALPHA} hd{STU_DIM}: {e}")