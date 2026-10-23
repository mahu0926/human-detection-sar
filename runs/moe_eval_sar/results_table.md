| Test set | Model | Precision | Recall | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| Heridal (vegetation)<br>159 img, 288 persons | All 3 experts + NMS | 0.782 | 0.670 | 0.709 | 0.328 |
|  | MoE (ours) | **0.803** | **0.781** | **0.801** | **0.369** |
| SeaDronesSee (maritime)<br>1,047 img, 4,372 persons | All 3 experts + NMS | 0.636 | 0.748 | 0.576 | 0.235 |
|  | MoE (ours) | **0.834** | **0.809** | **0.806** | **0.371** |
| NTIC (urban)<br>182 img, 1,367 persons | All 3 experts + NMS | 0.971 | 0.956 | 0.985 | 0.776 |
|  | MoE (ours) | **0.980** | **0.987** | **0.992** | **0.800** |
| Mean over domains | All 3 experts + NMS | | | 0.757 | 0.446 |
|  | MoE (ours) | | | **0.866** | **0.514** |
