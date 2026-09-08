### H3a. v7 is scored in-domain on the whole benchmark and still loses

| weight | split | n | p_recall | s_recall | p_mae_cond_s |
|---|---|---|---|---|---|
| jma_wc | all | 32144 | 0.881 [0.878, 0.885] | 0.549 [0.542, 0.556] | 0.239 |
| jma_wc | in_domain | 32144 | 0.881 [0.878, 0.885] | 0.549 [0.542, 0.556] | 0.239 |
| jma_wc | clean_holdout | 21864 | 0.909 [0.904, 0.912] | 0.527 [0.519, 0.536] | 0.250 |
| jma_wc_ft_global_v7 | all | 31992 | 0.853 [0.849, 0.857] | 0.505 [0.497, 0.512] | 0.198 |
| jma_wc_ft_global_v7 | in_domain | 31992 | 0.853 [0.849, 0.857] | 0.505 [0.497, 0.512] | 0.198 |
| jma_wc_ft_global_v7 | clean_holdout | 20827 | 0.876 [0.872, 0.880] | 0.477 [0.468, 0.485] | 0.211 |

For v7, in_domain == all: every one of the 12 benchmark datasets is in manifests_v2 (scripts/domain_registry.py:188-195 splits own models on the manifest's dataset set), so no cross_domain row exists for v7. The parent has no verifiable training split here (BASE_TRAINED_ON has no jma_wc entry), so its in_domain row is the fallback all-True mask. Contamination of the external sequences can only make v7 look better than it is; it cannot explain a loss.
