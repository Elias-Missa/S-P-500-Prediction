"""
Script to reorganize ML_Output folder structure by model type.

This script moves existing output folders into model-type subdirectories:
- LinearRegression -> regression/
- Ridge -> ridge/
- XGBoost -> xgboost/
- LSTM -> lstm/
- MLP -> mlp/
- Transformer -> transformer/
- RandomForest -> randomforest/
- CNN -> cnn/
- Ensemble -> ensemble/
"""

import os
import shutil

# Map model names to subdirectory names
MODEL_TYPE_MAP = {
    'LinearRegression': 'regression',
    'Ridge': 'ridge',
    'XGBoost': 'xgboost',
    'LSTM': 'lstm',
    'MLP': 'mlp',
    'Transformer': 'transformer',
    'RandomForest': 'randomforest',
    'CNN': 'cnn',
    'Ensemble': 'ensemble',
}

def reorganize_ml_output():
    """Reorganize existing ML_Output folders by model type."""
    # Get ML_Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_output_dir = os.path.join(script_dir, 'ML_Output')
    
    if not os.path.exists(ml_output_dir):
        print(f"ML_Output directory not found at: {ml_output_dir}")
        return
    
    # Get all folders in ML_Output
    all_items = os.listdir(ml_output_dir)
    folders = [item for item in all_items 
               if os.path.isdir(os.path.join(ml_output_dir, item))]
    
    moved_count = 0
    skipped_count = 0
    
    for folder in folders:
        # Skip if already in a model type subdirectory
        if folder in MODEL_TYPE_MAP.values():
            print(f"Skipping {folder}/ (already a model type directory)")
            skipped_count += 1
            continue
        
        # Extract model name from folder (format: ModelName_ordinal_...)
        parts = folder.split('_')
        if len(parts) < 2:
            print(f"Skipping {folder} (unexpected format)")
            skipped_count += 1
            continue
        
        model_name = parts[0]
        
        # Get target subdirectory
        if model_name in MODEL_TYPE_MAP:
            target_subdir = MODEL_TYPE_MAP[model_name]
        else:
            # Use lowercase model name as fallback
            target_subdir = model_name.lower()
            print(f"Warning: {model_name} not in map, using '{target_subdir}' as subdirectory")
        
        # Create target subdirectory if it doesn't exist
        target_dir = os.path.join(ml_output_dir, target_subdir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"Created subdirectory: {target_subdir}/")
        
        # Move folder
        source_path = os.path.join(ml_output_dir, folder)
        dest_path = os.path.join(target_dir, folder)
        
        if os.path.exists(dest_path):
            print(f"Skipping {folder} (destination already exists)")
            skipped_count += 1
        else:
            try:
                shutil.move(source_path, dest_path)
                print(f"Moved: {folder} -> {target_subdir}/{folder}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {folder}: {e}")
                skipped_count += 1
    
    print(f"\nReorganization complete!")
    print(f"  Moved: {moved_count} folders")
    print(f"  Skipped: {skipped_count} folders")

if __name__ == "__main__":
    print("Reorganizing ML_Output folder structure...")
    print("=" * 60)
    reorganize_ml_output()

