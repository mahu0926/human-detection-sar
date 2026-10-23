| Test set | Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| Heridal (vegetation)<br>157 img, 970 persons | All 3 experts + NMS | **0.874** | 0.858 | 0.906 | 0.520 |
|  | MoE (ours) | 0.872 | **0.894** | **0.927** | **0.542** |
| SeaDronesSee (maritime)<br>774 img, 3,489 persons | All 3 experts + NMS | 0.702 | 0.816 | 0.688 | 0.273 |
|  | MoE (ours) | **0.869** | **0.827** | **0.832** | **0.359** |
| NTIC (urban)<br>365 img, 2,755 persons | All 3 experts + NMS | 0.977 | 0.968 | 0.990 | 0.809 |
|  | MoE (ours) | **0.986** | **0.988** | **0.993** | **0.829** |
| Mean over domains | All 3 experts + NMS | | | 0.862 | 0.534 |
|  | MoE (ours) | | | **0.918** | **0.576** |
