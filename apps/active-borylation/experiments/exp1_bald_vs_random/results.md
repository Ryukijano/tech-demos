# exp1_bald_vs_random

**Hypothesis:** BALD (ensemble mutual information) acquisition over substrates reaches the target test MCC with fewer substrate molecules sampled than random acquisition.

**Metric:** test_mcc; target = 0.9 * MCC of same ensemble trained on full pool (per seed)

**Pass rule (fixed before running):** molecules_to_target(bald) < molecules_to_target(random) on >=2 of 3 seeds AND mean over seeds lower

|   seed |   full_pool_mcc |   target_mcc |   bald_molecules_to_target |   bald_final_mcc |   bald_auc_mcc |   random_molecules_to_target |   random_final_mcc |   random_auc_mcc | bald_better   |
|-------:|----------------:|-------------:|---------------------------:|-----------------:|---------------:|-----------------------------:|-------------------:|-----------------:|:--------------|
|      0 |          0.7064 |       0.6358 |                         60 |           0.6929 |         0.6    |                           20 |             0.6946 |           0.6553 | False         |
|      1 |          0.6964 |       0.6267 |                         90 |           0.6938 |         0.5588 |                           80 |             0.8304 |           0.6888 | False         |
|      2 |          0.6346 |       0.5711 |                         10 |           0.6486 |         0.6745 |                           10 |             0.719  |           0.6645 | False         |

Mean molecules to target: BALD 53.3, random 36.7. BALD better on 0/3 seeds.

**Result: FAIL**
