### H4b. Where the misses live: SNR of the traces each model misses, and of the traces only one detects

| phase | n | share_below_5dB_all | share_below_5dB_of_v7_misses | share_below_5dB_of_parent_misses | n_parent_only | share_below_5dB_of_parent_only | n_v7_only | share_below_5dB_of_v7_only | median_snr_parent_only_dB | median_snr_v7_only_dB |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 31880 | 0.332 [0.327, 0.337] | 0.707 [0.694, 0.719] | 0.696 [0.681, 0.712] | 1457 | 0.582 [0.557, 0.608] | 515 | 0.276 [0.237, 0.313] | 3.510 | 12.300 |
| S | 16967 | 0.241 [0.234, 0.247] | 0.232 [0.222, 0.241] | 0.222 [0.213, 0.233] | 936 | 0.346 [0.316, 0.376] | 195 | 0.415 [0.349, 0.482] | 9.060 | 7.080 |

The v13 config header reports '82% of v7's misses below 5 dB'; this table checks it on the committed results.
