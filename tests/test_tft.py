
import torch
from ML.tft_model import TFTModel

def test_tft():
    print("Testing TFT Model...")
    
    batch_size = 32
    seq_len = 10
    num_features = 20
    output_dim = 1
    
    # Mock Input
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Instantiate Model
    model = TFTModel(input_dim=num_features, hidden_dim=64, num_heads=4, num_layers=2, output_dim=output_dim)
    
    # Forward Pass
    try:
        y_pred = model(x)
        print(f"Forward pass successful. Output shape: {y_pred.shape}")
        
        assert y_pred.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {y_pred.shape}"
        print("Shape assertion passed.")
        
        # Check feature importances
        if model.feature_importances_ is not None:
            print(f"Feature importances shape: {model.feature_importances_.shape}")
            print("Feature importances capture successful.")
        else:
            print("Warning: feature_importances_ is None")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tft()
