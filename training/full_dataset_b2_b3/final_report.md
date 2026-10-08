# Full Dataset 8,486 Images - Evaluation Report

| Architecture | Input Method | Resolution | Overall Acc | Macro-F1 | Veg→Flow | Flow→Veg | Params (M) | Train Time |
|---|---|---|---|---|---|---|---|---|
| **EFFICIENTNET_B2** | Resize | 384x384 | 99.84% | **0.9972** | 0 | 0 | 7.71M | 83.7 min |
| **EFFICIENTNET_B2** | Voting | 288x288 | 100.00% | **1.0000** | 0 | 0 | 7.71M | 42.2 min |
| **EFFICIENTNET_B3** | Resize | 384x384 | 100.00% | **1.0000** | 0 | 0 | 10.70M | 81.4 min |
| **EFFICIENTNET_B3** | Voting | 288x288 | 99.84% | **0.9987** | 2 | 0 | 10.70M | 39.2 min |
