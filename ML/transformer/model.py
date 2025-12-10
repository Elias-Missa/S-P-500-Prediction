"""
Transformer model for time-series regression.

Uses PyTorch's built-in TransformerEncoder with positional encodings
for sequence modeling of financial time series.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer sequences.
    
    Adds position information to the input embeddings so the transformer
    can distinguish between different positions in the sequence.
    """
    
    def __init__(self, model_dim: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create sinusoidal position encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        
        pe = torch.zeros(1, max_len, model_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, model_dim)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer encoder model for time-series regression.
    
    Architecture:
    1. Linear projection: input_dim -> model_dim
    2. Positional encoding
    3. Stack of TransformerEncoderLayers
    4. Global average pooling over sequence dimension
    5. Linear projection to output (regression target)
    
    Args:
        input_dim: Number of input features per time step
        model_dim: Dimension of the transformer model (d_model)
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network in encoder layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim
        
        # Input projection: project features to model dimension
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(model_dim, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input/output format: (batch, seq, feature)
            norm_first=True    # Pre-norm architecture (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(model_dim)
        )
        
        # Output projection: aggregate -> single regression output
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, 1) - regression prediction
        """
        # Project input features to model dimension
        # (batch, seq_len, input_dim) -> (batch, seq_len, model_dim)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        # (batch, seq_len, model_dim) -> (batch, seq_len, model_dim)
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        # (batch, seq_len, model_dim) -> (batch, model_dim)
        x = x.mean(dim=1)
        
        # Project to output
        # (batch, model_dim) -> (batch, 1)
        output = self.output_projection(x)
        
        return output

