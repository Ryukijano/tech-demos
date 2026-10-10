# exp1 extremes (seed 0, BALD only, 30-min wall limit per config)

| config    |   ensemble_size |   batch_molecules | outcome   |   iterations |   molecules_reached |   final_mcc |   wall_s |   peak_rss_mb |
|:----------|----------------:|------------------:|:----------|-------------:|--------------------:|------------:|---------:|--------------:|
| ens10_b1  |              10 |                 1 | RUNNING   |          nan |                 nan |    nan      |    nan   |           nan |
| ens25_b10 |              25 |                10 | OK        |           25 |                 250 |      0.6832 |    348.3 |           315 |
| ens50_b10 |              50 |                10 | RUNNING   |          nan |                 nan |    nan      |    nan   |           nan |
| ens10_b25 |              10 |                25 | OK        |           11 |                 260 |      0.6971 |     51.7 |           306 |
| ens10_b50 |              10 |                50 | OK        |            6 |                 260 |      0.7203 |     25.6 |           305 |
