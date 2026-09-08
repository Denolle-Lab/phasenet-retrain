### Task 2d. Noise-pool detection MCC at each model's own best threshold

| weight | best_threshold | precision | recall | detection_mcc | mcc_ci_lo | mcc_ci_hi | source |
|---|---|---|---|---|---|---|---|
| jma_wc | 0.500 | 0.834 | 0.800 | 0.776 | 0.771 | 0.780 | paper_draft.qmd sec-detection table of 2026-08-10, transcribed |
| jma_wc_ft_global_v7 | 0.450 | 0.842 | 0.781 | 0.771 | 0.766 | 0.776 | paper_draft.qmd sec-detection table of 2026-08-10, transcribed |
| jma_wc_ft_global_v7_eventclean | 0.450 | 0.799 | 0.780 | 0.745 | 0.740 | 0.750 | paper_draft.qmd sec-detection table of 2026-08-10, transcribed |

Positive population: clean_holdout P arrivals; negative population: ~94k pure-noise traces (scripts/compute_detection_metrics.py). Not recomputed here: results/noise_fp_audit.csv is on the server.
