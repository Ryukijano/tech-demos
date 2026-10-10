# exp2_regio_inference

|   seed | set        | smiles              |   top1_idx |   top1_p |   labelled_sites | top1_correct   |   run_wall_s |   peak_rss_mb |
|-------:|:-----------|:--------------------|-----------:|---------:|-----------------:|:---------------|-------------:|--------------:|
|      0 | labelled   | COc1cc(OC)cc(OC)c1  |          7 |   0.9394 |                7 | True           |         19   |           820 |
|      0 | labelled   | CNC(=O)c1ccc(Br)cc1 |         10 |   0.929  |               10 | True           |         19   |           820 |
|      0 | labelled   | COc1cc(C)ccc1O      |          7 |   0.9969 |                7 | True           |         19   |           820 |
|      1 | labelled   | COc1cc(OC)cc(OC)c1  |          7 |   0.9394 |                7 | True           |         12.6 |           818 |
|      1 | labelled   | CNC(=O)c1ccc(Br)cc1 |         10 |   0.929  |               10 | True           |         12.6 |           818 |
|      1 | labelled   | COc1cc(C)ccc1O      |          7 |   0.9969 |                7 | True           |         12.6 |           818 |
|      2 | labelled   | COc1cc(OC)cc(OC)c1  |          7 |   0.9394 |                7 | True           |         10.7 |           819 |
|      2 | labelled   | CNC(=O)c1ccc(Br)cc1 |         10 |   0.929  |               10 | True           |         10.7 |           819 |
|      2 | labelled   | COc1cc(C)ccc1O      |          7 |   0.9969 |                7 | True           |         10.7 |           819 |
|      0 | unlabelled | c1ccc2ncccc2c1      |          6 |   0.9969 |                  |                |         75.1 |           790 |
|      0 | unlabelled | CC(=O)Nc1ccc(F)cc1  |          9 |   0.8621 |                  |                |         75.1 |           790 |
|      0 | unlabelled | Cn1ccc2ccccc21      |          2 |   0.9948 |                  |                |         75.1 |           790 |
|      1 | unlabelled | c1ccc2ncccc2c1      |          6 |   0.9969 |                  |                |          8.7 |           825 |
|      1 | unlabelled | CC(=O)Nc1ccc(F)cc1  |          9 |   0.8621 |                  |                |          8.7 |           825 |
|      1 | unlabelled | Cn1ccc2ccccc21      |          2 |   0.9948 |                  |                |          8.7 |           825 |
|      2 | unlabelled | c1ccc2ncccc2c1      |          6 |   0.9969 |                  |                |          3.4 |           816 |
|      2 | unlabelled | CC(=O)Nc1ccc(F)cc1  |          9 |   0.8621 |                  |                |          3.4 |           816 |
|      2 | unlabelled | Cn1ccc2ccccc21      |          2 |   0.9948 |                  |                |          3.4 |           816 |

top1_site_accuracy per seed: {0: 1.0, 1: 1.0, 2: 1.0}

max wall_s per molecule (incl. model load): 25.0

**Result: PASS**

Caveat: all 3 labelled example molecules also appear in the upstream regioselectivity training set, so this is a load-and-run sanity check, not a generalisation test.
