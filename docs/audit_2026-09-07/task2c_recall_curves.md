### Task 2c. Recall curves, all distances (the budget on positives equals recall x n)

| phase | threshold | n | recall_parent | picks_parent | recall_v7 | picks_v7 | threshold_v7_for_parent_budget |
|---|---|---|---|---|---|---|---|
| P | 0.100 | 31880 | 0.938 | 29895 | 0.927 | 29568 | 0.071 |
| P | 0.200 | 31880 | 0.915 | 29170 | 0.893 | 28459 | 0.135 |
| P | 0.300 | 31880 | 0.884 | 28174 | 0.854 | 27232 | 0.224 |
| P | 0.400 | 31880 | 0.843 | 26870 | 0.802 | 25562 | 0.325 |
| P | 0.500 | 31880 | 0.782 | 24927 | 0.732 | 23332 | 0.431 |
| P | 0.600 | 31880 | 0.704 | 22432 | 0.648 | 20670 | 0.536 |
| P | 0.700 | 31880 | 0.602 | 19191 | 0.554 | 17652 | 0.653 |
| P | 0.800 | 31880 | 0.476 | 15182 | 0.434 | 13837 | 0.767 |
| P | 0.900 | 31880 | 0.306 | 9751 | 0.265 | 8452 | 0.880 |
| S | 0.100 | 16967 | 0.629 | 10674 | 0.613 | 10404 | 0.070 |
| S | 0.200 | 16967 | 0.593 | 10062 | 0.562 | 9539 | 0.139 |
| S | 0.300 | 16967 | 0.549 | 9314 | 0.505 | 8573 | 0.226 |
| S | 0.400 | 16967 | 0.485 | 8222 | 0.446 | 7563 | 0.334 |
| S | 0.500 | 16967 | 0.409 | 6948 | 0.384 | 6507 | 0.459 |
| S | 0.600 | 16967 | 0.325 | 5521 | 0.308 | 5220 | 0.581 |
| S | 0.700 | 16967 | 0.231 | 3920 | 0.224 | 3792 | 0.691 |
| S | 0.800 | 16967 | 0.128 | 2169 | 0.139 | 2364 | 0.813 |
| S | 0.900 | 16967 | 0.036 | 617 | 0.049 | 827 | 0.914 |

`threshold_v7_for_parent_budget` is the v7 threshold that emits the parent's number of picks on these positives; by construction it gives v7 the parent's recall, which is why matched-budget recall is uninformative here and must come from continuous data with false alarms.
