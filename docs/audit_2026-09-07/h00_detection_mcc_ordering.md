### H00. Ordering under the noise-pool detection MCC versus the metrics that drove selection

| weight | detection_mcc | p_mae_unconditional_all | p_recall_all | p_recall_clean_holdout | p_auc_recall_all | rank_detection_mcc | rank_p_mae_selection_metric | rank_p_recall_all | rank_p_auc_recall |
|---|---|---|---|---|---|---|---|---|---|
| jma_wc_ft_ensemble_v7v11 | 0.823 [0.819, 0.827] | 0.339 | 0.834 | 0.856 | 0.653 | 1 | 1 | 7 | 5 |
| jma_wc_ft_global_v11 | 0.804 [0.799, 0.808] | 0.364 | 0.810 | 0.828 | 0.667 | 2 | 5 | 9 | 2 |
| jma_wc_ft_ensemble_v3v7 | 0.780 [0.776, 0.785] | 0.356 | 0.857 | 0.881 | 0.648 | 3 | 4 | 5 | 7 |
| jma_wc | 0.776 [0.771, 0.780] | 0.374 | 0.881 | 0.909 | 0.693 | 4 | 7 | 2 | 1 |
| jma_wc_ft_global_v7 | 0.771 [0.766, 0.776] | 0.340 | 0.853 | 0.876 | 0.660 | 5 | 2 | 6 | 3 |
| jma_wc_ft_global_v7_eventclean | 0.745 [0.740, 0.750] | 0.344 | 0.862 | 0.881 | 0.656 | 6 | 3 | 4 | 4 |
| jma_wc_ft_global_v20 | 0.671 [0.666, 0.676] | 0.382 | 0.833 | 0.843 | 0.586 | 7 | 8 | 8 | 9 |
| jma_wc_ft_global_v3 | 0.648 [0.642, 0.654] | 0.368 | 0.872 | 0.885 | 0.649 | 8 | 6 | 3 | 6 |
| jma_wc_ft_global_v18 | 0.567 [0.561, 0.574] | 0.433 | 0.896 | 0.888 | 0.607 | 9 | 9 | 1 | 8 |

Spearman(rank detection MCC, rank_p_mae_selection_metric) = 0.65 over 9 weights; Spearman(rank detection MCC, rank_p_recall_all) = -0.57 over 9 weights; Spearman(rank detection MCC, rank_p_auc_recall) = 0.52 over 9 weights.

Source: the 2026-08-10 table transcribed from paper_draft.qmd. Only 9 own weights were ever run over the noise pool. Never scored: jma_wc_ft_global_v2, jma_wc_ft_global_v4, jma_wc_ft_global_v5, jma_wc_ft_global_v6, jma_wc_ft_global_v8, jma_wc_ft_global_v9, jma_wc_ft_global_v10, jma_wc_ft_global_v12, jma_wc_ft_global_v13, jma_wc_ft_global_v14, jma_wc_ft_global_v15, jma_wc_ft_global_v16, jma_wc_ft_global_v17, jma_wc_ft_global_v19, jma_wc_ft, jma_wc_ft_frozen, jma_wc_ft_global, jma_wc_ft_noise (their checkpoints, if kept, are under checkpoints/ on the server; scripts/audit_noise_fp_leaderboard.py SINGLE_MODELS lists what was run). The full v1-v20 ordering H00 asks for cannot be produced without them.
