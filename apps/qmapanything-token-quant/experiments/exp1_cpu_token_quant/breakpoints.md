# Breakpoints (seed 0, not part of the PASS rule)

## Lower bits

| W | A | method | special_rel | patch_rel | metric_rel |
|---:|---:|---|---:|---:|---:|
| 3 | 3 | naive | 1.0038 | 0.8642 | 0.8756 |
| 3 | 3 | rotation | 0.9507 | 0.6155 | 0.3606 |
| 3 | 3 | qmap | 0.9077 | 0.6191 | 0.3944 |
| 2 | 2 | naive | 1.0009 | 1.0058 | 0.8746 |
| 2 | 2 | rotation | 1.0009 | 0.9875 | 0.8746 |
| 2 | 2 | qmap | 0.7046 | 0.7344 | 0.3544 |
| 4 | 2 | naive | 1.0009 | 0.9692 | 0.8746 |
| 4 | 2 | rotation | 1.0009 | 0.9543 | 0.8746 |
| 4 | 2 | qmap | 0.6133 | 0.6857 | 0.3587 |

## More patch tokens (real qkv rows, per-scene absmax, 4-bit activations)

| patches | special naive | patch naive | special rotation | patch rotation | delta shrink |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.2485 | 0.4934 | 0.4150 | 0.1389 | 3.727 |
| 64 | 0.3291 | 0.5880 | 0.4664 | 0.1527 | 4.201 |
| 256 | 0.3830 | 0.6347 | 0.5713 | 0.1692 | 4.255 |
| 1024 | 0.4791 | 0.6686 | 0.6137 | 0.1769 | 4.521 |

## Matched max-abs (grid deficit removed)

- original special/patch naive: 0.822
- matched special/patch naive: 0.374
- original held-out special MSE drop: 0.016
- matched held-out special MSE drop: 0.068
- original 1/kappa L2: 6.020
- matched 1/kappa L2: 3.000

## One global block, qmap / rotation latency

| tokens | qmap/rotation |
|---:|---:|
| 33 | 1.1487 |
| 128 | 1.1452 |
| 512 | 1.0856 |
