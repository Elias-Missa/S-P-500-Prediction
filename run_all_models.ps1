# Run all 3 models sequentially

Write-Host "Starting Ridge Run..."
python -m ML.train_walkforward --model_type Ridge

Write-Host "Starting RegimeGatedRidge Run..."
python -m ML.train_walkforward --model_type RegimeGatedRidge

Write-Host "Starting RegimeGatedHybrid (DualVolRidgeTree) Run..."
python -m ML.train_walkforward --model_type RegimeGatedHybrid

Write-Host "All runs complete."
