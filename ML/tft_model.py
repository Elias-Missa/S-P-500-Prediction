import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU).
    Input: (..., input_dim)
    Output: (..., input_dim / 2) if dim is doubled in linear, else needs check.
    Usually GLU(x) = x * sigmoid(gate(x)) ??
    No, standard GLU is GLU(x) = linear(x) * sigmoid(linear_gate(x)) or similar.
    Paper: GLU(x) = \sigma(W_2 x + b_2) \odot (W_1 x + b_1)
    """
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.fc = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x):
        # x: (..., input_dim)
        x = self.fc(x)
        # Split into value and gate
        val, gate = x.chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """
    GRN: Gated Residual Network.
    x -> [Linear -> ELU -> Linear -> Dropout -> GLU] + x -> LayerNorm
    Also takes an optional context c.
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
            
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Project residual if dims mismatch
        if input_dim != hidden_dim:
            self.res_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.res_proj = None

    def forward(self, x, context=None):
        # x: (..., input_dim)
        residual = x
        if self.res_proj is not None:
            residual = self.res_proj(residual)
            
        x = self.fc1(x)
        if context is not None and self.context_fc is not None:
            # context: (..., context_dim) -> broadcast or align?
            # Assuming context is (Batch, ContextDim) and x is (Batch, Seq, InputDim)
            # We need to unsqueeze context: (Batch, 1, Hidden)
            c = self.context_fc(context)
            if c.dim() == 2 and x.dim() == 3:
                c = c.unsqueeze(1)
            x = x + c
            
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        
        return self.norm(x + residual)


class VariableSelectionNetwork(nn.Module):
    """
    VSN: Selects relevant features.
    Architecture:
    1. Apply GRN to each feature INDIVIDUALLY.
    2. Calculate weights via a GRN over the flattened features (or transformed representations).
    3. Weighted sum of features.
    """
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Independent GRNs for each feature
        # Since we have continuous inputs, we first project each scalar feature to hidden_dim
        # then apply a GRN.
        # Optimization: We can share weights or not? TFT paper says "variable selection network"
        # usually implies transforming inputs to a common space first.
        # We will use Linear(1, hidden) -> GRN for each feature.
        
        self.feature_encoders = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, dropout) for _ in range(num_features)
        ])
        
        # Selection GRN
        # Takes concatenation of state_h (from encoders) or just flat inputs?
        # Paper: "flattened vector of all inputs ... is fed into a Gated Residual Network" to get weights
        # Actually it computes weights based on the state vectors.
        # Let's simplify: We concat the transformed features, then compute weights?
        # Re-reading standard VSN:
        # v_xt = Flatten(GRN_xi(xi)) or just concat... 
        # Weights = Softmax(GRN_weights(concat(v_xt)))
        # Here we will concat the "state" of each feature.
        
        # To avoid massive param count for 'GRN_weights' if num_features is large, 
        # we can just use a simple MLP for weights calculation.
        
        self.selection_grn = GatedResidualNetwork(num_features * hidden_dim, num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (Batch, Seq, NumFeatures)
        # Process each feature
        # We need to unsqueeze the feature dimension to apply GRN(1 -> Hidden)
        
        batch_size, seq_len, _ = x.shape
        
        transformed_features = []
        for i in range(self.num_features):
            feat = x[..., i:i+1] # (Batch, Seq, 1)
            enc = self.feature_encoders[i](feat) # (Batch, Seq, Hidden)
            transformed_features.append(enc)
            
        # Stack: (Batch, Seq, NumFeatures, Hidden)
        # But for weight calculation we flatten features
        
        concat_features = torch.cat(transformed_features, dim=-1) # (Batch, Seq, NumFeatures * Hidden)
        
        # Selection Weights
        # GRN outputs (Batch, Seq, NumFeatures)
        selection_weights = self.selection_grn(concat_features) 
        selection_weights = self.softmax(selection_weights) # (Batch, Seq, NumFeatures)
        
        # Weighted Sum
        # weights: (Batch, Seq, NumFeatures) -> unsqueeze -> (Batch, Seq, NumFeatures, 1)
        # transformed: (Batch, Seq, NumFeatures, Hidden) -- wait, we need to stack properly first
        
        # Reshape transformed for weighting
        # We have list of (Batch, Seq, Hidden). Stack dim 2.
        stacked_features = torch.stack(transformed_features, dim=2) # (Batch, Seq, NumFeatures, Hidden)
        
        weights_expanded = selection_weights.unsqueeze(-1) # (Batch, Seq, NumFeatures, 1)
        
        weighted_sum = torch.sum(stacked_features * weights_expanded, dim=2) # (Batch, Seq, Hidden)
        
        return weighted_sum, selection_weights


class TFTModel(nn.Module):
    """
    Temporal Fusion Transformer (Lite Version).
    
    Architecture:
    1. Variable Selection Network (VSN) -> Feature Importance
    2. LSTM Encoder (processing sequence)
    3. Gated Residual Network (post-LSTM)
    4. Multi-Head Attention (Self-Attention on History) - Optional or just skip to output for "Lite"
    5. Output Layer
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1, output_dim=1):
        super().__init__()
        
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        
        # LSTM for temporal processing (Key part of TFT uses LSTM for local processing)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Gating after LSTM
        self.post_lstm_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # Multi-Head Attention (Interpretability: "Where to look")
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.post_attn_gate = GatedResidualNetwork(hidden_dim, hidden_dim, dropout)
        
        # Final Output
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        
        self.feature_importances_ = None # Placeholder for interpretation
        
    def forward(self, x):
        # x: (Batch, Seq, Features)
        
        # 1. Variable Selection
        x_vsn, selection_weights = self.vsn(x) 
        # x_vsn: (Batch, Seq, Hidden)
        # selection_weights: (Batch, Seq, NumFeatures)
        
        # Store latest variable selection weights for interpretability (average over batch/seq)
        # This is a hacky way to expose it to sklearn-style access, typically we'd return it.
        # We'll store the mean over the batch and sequence (last batch)
        with torch.no_grad():
            self.feature_importances_ = selection_weights.mean(dim=(0, 1)).cpu().numpy()
            
        # 2. LSTM (Local temporal processing)
        # TFT typically uses LSTM for Encoder/Decoder. Here we just encode history.
        x_lstm, _ = self.lstm(x_vsn)
        x_lstm = self.post_lstm_gate(x_lstm + x_vsn) # Skip connection from VSN
        
        # 3. Attention (Global temporal dependency)
        # Self-attention on the LSTM output
        # Query = Key = Value = x_lstm
        attn_output, attn_weights = self.attention(x_lstm, x_lstm, x_lstm)
        x_attn = self.post_attn_gate(attn_output + x_lstm) # Skip connection
        
        # 4. Final Prediction (Project last step)
        # We take the LAST time step for prediction
        last_step = x_attn[:, -1, :] # (Batch, Hidden)
        output = self.output_fc(last_step)
        
        return output
