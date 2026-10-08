$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$csvDir = "e:\paddy_test\training\full_dataset_b2_b3"

Write-Host "======================================================"
Write-Host "1. F1 -> F2 (Test F2) [4 configs, seed 42]"
Write-Host "======================================================"
python 03_train_resizing_384.py --split-csv "$csvDir\split_test_F2_noval.csv" --epochs 5 --arch all --seed 42
python 04_train_voting_288.py --split-csv "$csvDir\split_test_F2_noval.csv" --epochs 5 --arch all --seed 42

Write-Host "======================================================"
Write-Host "2. F2 -> F1 (Test F1) [4 configs, seed 42]"
Write-Host "======================================================"
python 03_train_resizing_384.py --split-csv "$csvDir\split_test_F1_noval.csv" --epochs 5 --arch all --seed 42
python 04_train_voting_288.py --split-csv "$csvDir\split_test_F1_noval.csv" --epochs 5 --arch all --seed 42

Write-Host "======================================================"
Write-Host "3. F1 -> F2 (Test F2) [B2-Resize, seeds 43 & 44]"
Write-Host "======================================================"
python 03_train_resizing_384.py --split-csv "$csvDir\split_test_F2_noval.csv" --epochs 5 --arch efficientnet_b2 --seed 43
python 03_train_resizing_384.py --split-csv "$csvDir\split_test_F2_noval.csv" --epochs 5 --arch efficientnet_b2 --seed 44

Write-Host "======================================================"
Write-Host "4. Controls [B2-Resize, seed 42]"
Write-Host "======================================================"
python 03_train_resizing_384.py --split-csv "$csvDir\split_control_F1.csv" --epochs 5 --arch efficientnet_b2 --seed 42
python 03_train_resizing_384.py --split-csv "$csvDir\split_control_F2.csv" --epochs 5 --arch efficientnet_b2 --seed 42

Write-Host "======================================================"
Write-Host "ALL PIPELINE RUNS COMPLETED SUCCESSFULLY"
Write-Host "======================================================"
