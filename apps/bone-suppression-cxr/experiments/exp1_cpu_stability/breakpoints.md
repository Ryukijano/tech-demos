# Breakpoints (seed 0, bone model; SSIM vs clean bone output)

ok >= 0.95, degraded >= 0.8, broken < 0.8. Rotation compares with the clean output rotated by the same angle.

| image | family | level | ssim_bone | psnr_bone | bone_mean | looks_inverted | verdict |
|---|---|---|---|---|---|---|---|
| JPCNN001 | noise_sigma | 0.01 | 0.9606 | 35.2 | 0.0274 | False | ok |
| JPCNN001 | noise_sigma | 0.03 | 0.8751 | 29.3 | 0.0262 | False | degraded |
| JPCNN001 | noise_sigma | 0.05 | 0.7892 | 25.7 | 0.0228 | False | broken |
| JPCNN001 | noise_sigma | 0.1 | 0.6371 | 21.3 | 0.0164 | False | broken |
| JPCNN001 | noise_sigma | 0.2 | 0.4999 | 18.5 | 0.0089 | False | broken |
| JPCNN001 | noise_sigma | 0.4 | 0.4082 | 17.0 | 0.0032 | False | broken |
| JPCNN001 | lowres_px | 512 | 0.9866 | 37.9 | 0.0302 | False | ok |
| JPCNN001 | lowres_px | 256 | 0.9318 | 31.7 | 0.0310 | False | degraded |
| JPCNN001 | lowres_px | 128 | 0.8083 | 26.5 | 0.0290 | False | degraded |
| JPCNN001 | lowres_px | 64 | 0.4543 | 17.5 | 0.0056 | False | broken |
| JPCNN001 | gamma | 0.33 | 0.8231 | 21.9 | 0.0144 | False | degraded |
| JPCNN001 | gamma | 0.5 | 0.9231 | 25.2 | 0.0193 | False | degraded |
| JPCNN001 | gamma | 2.0 | 0.9202 | 22.6 | 0.0356 | False | degraded |
| JPCNN001 | gamma | 3.0 | 0.8052 | 18.2 | 0.0373 | False | degraded |
| JPCNN001 | contrast | 0.25 | 1.0000 | 131.3 | 0.0285 | False | ok |
| JPCNN001 | contrast | 3.0 | 0.4465 | 13.8 | 0.0329 | False | broken |
| JPCNN001 | rotation_deg | 5 | 0.9821 | 34.9 | 0.0283 | False | ok |
| JPCNN001 | rotation_deg | 15 | 0.9776 | 32.7 | 0.0271 | False | ok |
| JPCNN001 | rotation_deg | 30 | 0.9707 | 30.9 | 0.0257 | False | ok |
| JPCNN001 | rotation_deg | 90 | 0.9755 | 37.4 | 0.0283 | False | ok |
| JPCNN001 | jpeg_quality | 30 | 0.8232 | 24.9 | 0.0324 | False | degraded |
| JPCNN001 | jpeg_quality | 10 | 0.6444 | 20.7 | 0.0361 | False | broken |
| JPCNN001 | jpeg_quality | 3 | 0.3932 | 13.3 | 0.0399 | False | broken |
| JPCNN001 | inverted | True | 0.2496 | 13.6 | 0.0316 | True | broken |
| cxr_train_1_a_1 | noise_sigma | 0.01 | 0.9554 | 30.1 | 0.0257 | False | ok |
| cxr_train_1_a_1 | noise_sigma | 0.03 | 0.8961 | 27.4 | 0.0225 | False | degraded |
| cxr_train_1_a_1 | noise_sigma | 0.05 | 0.8299 | 24.3 | 0.0190 | False | degraded |
| cxr_train_1_a_1 | noise_sigma | 0.1 | 0.6683 | 19.7 | 0.0114 | False | broken |
| cxr_train_1_a_1 | noise_sigma | 0.2 | 0.5195 | 17.1 | 0.0052 | False | broken |
| cxr_train_1_a_1 | noise_sigma | 0.4 | 0.4780 | 16.2 | 0.0019 | False | broken |
| cxr_train_1_a_1 | lowres_px | 512 | 0.9686 | 30.8 | 0.0279 | False | ok |
| cxr_train_1_a_1 | lowres_px | 256 | 0.9661 | 30.3 | 0.0265 | False | ok |
| cxr_train_1_a_1 | lowres_px | 128 | 0.8708 | 25.3 | 0.0273 | False | degraded |
| cxr_train_1_a_1 | lowres_px | 64 | 0.4683 | 16.0 | 0.0013 | False | broken |
| cxr_train_1_a_1 | gamma | 0.33 | 0.8175 | 20.2 | 0.0132 | False | degraded |
| cxr_train_1_a_1 | gamma | 0.5 | 0.9127 | 22.4 | 0.0181 | False | degraded |
| cxr_train_1_a_1 | gamma | 2.0 | 0.9240 | 21.7 | 0.0360 | False | degraded |
| cxr_train_1_a_1 | gamma | 3.0 | 0.8451 | 17.7 | 0.0391 | False | degraded |
| cxr_train_1_a_1 | contrast | 0.25 | 1.0000 | 125.7 | 0.0270 | False | ok |
| cxr_train_1_a_1 | contrast | 3.0 | 0.5369 | 13.7 | 0.0291 | False | broken |
| cxr_train_1_a_1 | rotation_deg | 5 | 0.9617 | 29.7 | 0.0280 | False | ok |
| cxr_train_1_a_1 | rotation_deg | 15 | 0.9739 | 32.8 | 0.0268 | False | ok |
| cxr_train_1_a_1 | rotation_deg | 30 | 0.9616 | 31.0 | 0.0265 | False | ok |
| cxr_train_1_a_1 | rotation_deg | 90 | 0.9716 | 30.9 | 0.0280 | False | ok |
| cxr_train_1_a_1 | jpeg_quality | 30 | 0.8835 | 23.8 | 0.0300 | False | degraded |
| cxr_train_1_a_1 | jpeg_quality | 10 | 0.7072 | 21.0 | 0.0294 | False | broken |
| cxr_train_1_a_1 | jpeg_quality | 3 | 0.4890 | 14.4 | 0.0412 | False | broken |
| cxr_train_1_a_1 | inverted | True | 0.3376 | 15.4 | 0.0210 | True | broken |

## First level that leaves 'ok'

- JPCNN001 / noise_sigma: 0.03 -> degraded (SSIM 0.875)
- JPCNN001 / lowres_px: 256 -> degraded (SSIM 0.932)
- JPCNN001 / gamma: 0.33 -> degraded (SSIM 0.823)
- JPCNN001 / contrast: 3.0 -> broken (SSIM 0.447)
- JPCNN001 / rotation_deg: never (all levels ok)
- JPCNN001 / jpeg_quality: 30 -> degraded (SSIM 0.823)
- JPCNN001 / inverted: True -> broken (SSIM 0.250)
- cxr_train_1_a_1 / noise_sigma: 0.03 -> degraded (SSIM 0.896)
- cxr_train_1_a_1 / lowres_px: 128 -> degraded (SSIM 0.871)
- cxr_train_1_a_1 / gamma: 0.33 -> degraded (SSIM 0.818)
- cxr_train_1_a_1 / contrast: 3.0 -> broken (SSIM 0.537)
- cxr_train_1_a_1 / rotation_deg: never (all levels ok)
- cxr_train_1_a_1 / jpeg_quality: 30 -> degraded (SSIM 0.884)
- cxr_train_1_a_1 / inverted: True -> broken (SSIM 0.338)
