# BNS Retrieval Evaluation — Final Walkthrough

## Alpha Sensitivity Results

| α | Acc@3 | Acc@5 | MRR@5 | Direct | Paraphrased | Scenario | Multi-sect | Confusing |
|---|---|---|---|---|---|---|---|---|
| 0.3 (BM25 heavy) | 63.67% | 69.33% | 0.5330 | 71.67% | 96.67% | 53.33% | 71.67% | 53.33% |
| **0.5 (balanced)** | **68.33%** | **73.67%** | 0.5769 | 80.00% | **100%** | 55.56% | **76.67%** | 56.67% |
| 0.7 (Dense heavy) | **70.67%** | 73.33% | **0.5956** | **85.00%** | **100%** | **56.67%** | **76.67%** | 40.00% |

> [!IMPORTANT]
> **α=0.7 wins @3** (70.67%) with best MRR, **α=0.5 wins @5** (73.67%). The dense-heavy config excels on Direct/Paraphrased but the balanced config is more robust on Confusing queries.

![Alpha vs Accuracy@5](C:/Users/Shreyas Durge/.gemini/antigravity/brain/b925036e-6604-4944-a98d-6b35fc62bffc/alpha_vs_accuracy.png)

![Category-wise Accuracy@5 by Alpha](C:/Users/Shreyas Durge/.gemini/antigravity/brain/b925036e-6604-4944-a98d-6b35fc62bffc/alpha_category_bars.png)

## Statistical Validation (Paired t-tests)

| Test | t-stat | p-value | Verdict |
|---|---|---|---|
| Dense vs Hybrid @3 | -2.7172 | **0.0070** | ✅✅ Highly Significant |
| Dense vs Hybrid @5 | -1.6078 | 0.1089 | ❌ Not Significant |
| Hybrid vs Rerank @3 | 3.7733 | **0.0002** | ✅✅ Highly Significant (Rerank worse) |
| Hybrid vs Rerank @5 | 1.2070 | 0.2284 | ❌ Not Significant |

> [!NOTE]
> At Top-3, Hybrid statistically significantly outperforms Dense (p=0.007) and the Cross-Encoder significantly degrades performance (p=0.0002). At Top-5, differences are not statistically significant.

## Full System Comparison

| System | @3 Acc | @5 Acc | @5 MRR |
|---|---|---|---|
| Dense (MPNet) | 61.33% | 69.67% | 0.5377 |
| **Hybrid (α=0.5)** | **68.33%** | **73.67%** | **0.5771** |
| Hybrid + CrossEncoder | 58.33% | 71.00% | 0.5027 |

## Execution

```powershell
conda run -n bns_rag python src/alpha_tuning_study.py
conda run -n bns_rag python src/statistical_validation.py
```
