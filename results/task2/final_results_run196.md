# Final test results: run196

Accuracy: normalized exact match (grounding) or VQA soft accuracy (VQAv2), in %. Brackets: 95% interval from 4,000 bootstrap resamples of complete pairs. NLL: mean reference-answer token NLL (end-of-turn excluded) under each image condition, always against the original references.


## Annotation-derived grounding (128 questions)

| Image condition | Accuracy | By type | Pair accuracy | Identical answers | Empty | Follows partner answer | Ref. NLL | Gen. s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 80.5 [72.7, 88.3] | counting 64.1, presence 96.9 | 68.8 [57.8, 79.7] | 20.3 | 0.0 | 14.1 | 0.470 | 0.040 |
| Complementary swap | 14.1 [8.6, 20.3] | counting 25.0, presence 3.1 | 3.1 [0.0, 7.8] | 20.3 | 0.0 | 80.5 | 3.534 | 0.037 |
| Mismatched | 40.6 [34.4, 47.7] | counting 29.7, presence 51.6 | 6.2 [1.6, 12.5] | 78.1 | 0.0 | 38.3 | 2.204 | 0.038 |
| No image | 43.0 [38.3, 46.9] | counting 35.9, presence 50.0 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 43.0 | 1.139 | 0.021 |

Correct-image advantage (correct minus other condition; NLL: other minus correct, so positive means the correct image makes the reference more likely):

| Versus | Δ accuracy (pp) | Δ NLL | Δ pair accuracy (pp) |
|---|---|---|---|
| Complementary swap | 66.4 [52.3, 78.9] | 3.064 [2.563, 3.598] | 65.6 [51.6, 78.1] |
| Mismatched | 39.8 [31.2, 48.4] | 1.734 [1.446, 2.013] | 62.5 [50.0, 75.0] |
| No image | 37.5 [29.7, 44.5] | 0.668 [0.525, 0.810] | 68.8 [56.2, 79.7] |

Reference NLL by question type:

| Type | Correct | Complementary swap | Mismatched | No image |
|---|---|---|---|---|
| counting | 0.870 | 2.605 | 2.186 | 1.346 |
| presence | 0.070 | 4.463 | 2.221 | 0.931 |

## Natural VQAv2 (192 questions)

| Image condition | Accuracy | By type | Pair accuracy | Identical answers | Empty | Follows partner answer | Ref. NLL | Gen. s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 53.7 [47.1, 60.1] | number 52.1, other 47.1, yes/no 61.4 | 19.8 [11.5, 28.1] | 49.0 | 0.0 | 27.1 | 1.748 | 0.043 |
| Complementary swap | 31.8 [25.5, 38.4] | number 20.8, other 21.0, yes/no 49.9 | 5.2 [1.0, 10.4] | 49.0 | 0.0 | 47.9 | 2.658 | 0.045 |
| Mismatched | 31.9 [25.9, 38.0] | number 16.9, other 22.9, yes/no 50.8 | 5.2 [1.0, 9.4] | 57.3 | 0.0 | 31.8 | 2.625 | 0.049 |
| No image | 35.8 [30.7, 41.0] | number 33.8, other 18.5, yes/no 54.6 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 31.8 | 2.273 | 0.028 |

Correct-image advantage (correct minus other condition; NLL: other minus correct, so positive means the correct image makes the reference more likely):

| Versus | Δ accuracy (pp) | Δ NLL | Δ pair accuracy (pp) |
|---|---|---|---|
| Complementary swap | 21.9 [12.6, 31.3] | 0.909 [0.523, 1.336] | 14.6 [5.2, 24.0] |
| Mismatched | 21.8 [14.5, 29.3] | 0.877 [0.533, 1.235] | 14.6 [6.2, 22.9] |
| No image | 17.9 [11.6, 24.7] | 0.525 [0.219, 0.838] | 19.8 [12.5, 28.1] |

Reference NLL by question type:

| Type | Correct | Complementary swap | Mismatched | No image |
|---|---|---|---|---|
| number | 1.987 | 3.688 | 3.356 | 2.174 |
| other | 2.515 | 3.437 | 3.686 | 3.588 |
| yes/no | 0.822 | 1.192 | 1.078 | 1.024 |
