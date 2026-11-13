# Final test results: run49

Accuracy: normalized exact match (grounding) or VQA soft accuracy (VQAv2), in %. Brackets: 95% interval from 4,000 bootstrap resamples of complete pairs. NLL: mean reference-answer token NLL (end-of-turn excluded) under each image condition, always against the original references.


## Annotation-derived grounding (128 questions)

| Image condition | Accuracy | By type | Pair accuracy | Identical answers | Empty | Follows partner answer | Ref. NLL | Gen. s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 77.3 [68.8, 85.2] | counting 60.9, presence 93.8 | 64.1 [51.6, 75.0] | 21.9 | 0.0 | 15.6 | 0.575 | 0.025 |
| Complementary swap | 15.6 [9.4, 22.7] | counting 25.0, presence 6.2 | 3.1 [0.0, 7.8] | 21.9 | 0.0 | 77.3 | 3.420 | 0.023 |
| Mismatched | 43.0 [37.5, 48.4] | counting 42.2, presence 43.8 | 4.7 [0.0, 10.9] | 71.9 | 0.0 | 40.6 | 1.896 | 0.023 |
| No image | 43.0 [38.3, 46.9] | counting 35.9, presence 50.0 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 43.0 | 1.209 | 0.023 |

Correct-image advantage (correct minus other condition; NLL: other minus correct, so positive means the correct image makes the reference more likely):

| Versus | Δ accuracy (pp) | Δ NLL | Δ pair accuracy (pp) |
|---|---|---|---|
| Complementary swap | 61.7 [47.7, 74.2] | 2.846 [2.302, 3.414] | 60.9 [46.9, 73.4] |
| Mismatched | 34.4 [25.0, 43.8] | 1.321 [1.009, 1.631] | 59.4 [46.9, 71.9] |
| No image | 34.4 [26.6, 41.4] | 0.635 [0.468, 0.795] | 64.1 [51.6, 75.0] |

Reference NLL by question type:

| Type | Correct | Complementary swap | Mismatched | No image |
|---|---|---|---|---|
| counting | 0.958 | 2.669 | 1.813 | 1.506 |
| presence | 0.191 | 4.172 | 1.979 | 0.913 |

## Natural VQAv2 (192 questions)

| Image condition | Accuracy | By type | Pair accuracy | Identical answers | Empty | Follows partner answer | Ref. NLL | Gen. s/answer |
|---|---|---|---|---|---|---|---|---|
| Correct | 53.5 [46.5, 60.3] | number 41.0, other 47.4, yes/no 67.9 | 22.9 [14.6, 31.3] | 54.2 | 0.0 | 26.6 | 1.668 | 0.029 |
| Complementary swap | 31.2 [25.3, 37.2] | number 26.5, other 21.5, yes/no 44.2 | 3.1 [0.0, 7.3] | 54.2 | 0.0 | 49.0 | 2.521 | 0.028 |
| Mismatched | 33.0 [27.4, 38.9] | number 25.2, other 21.1, yes/no 50.0 | 3.1 [0.0, 7.3] | 60.4 | 0.0 | 30.2 | 2.508 | 0.026 |
| No image | 34.0 [28.9, 39.0] | number 31.2, other 20.1, yes/no 49.6 | 0.0 [0.0, 0.0] | 100.0 | 0.0 | 30.2 | 2.170 | 0.027 |

Correct-image advantage (correct minus other condition; NLL: other minus correct, so positive means the correct image makes the reference more likely):

| Versus | Δ accuracy (pp) | Δ NLL | Δ pair accuracy (pp) |
|---|---|---|---|
| Complementary swap | 22.2 [12.6, 31.7] | 0.853 [0.509, 1.222] | 19.8 [10.4, 29.2] |
| Mismatched | 20.5 [12.1, 28.9] | 0.840 [0.512, 1.187] | 19.8 [10.4, 29.2] |
| No image | 19.5 [12.6, 26.6] | 0.502 [0.212, 0.792] | 22.9 [14.6, 32.3] |

Reference NLL by question type:

| Type | Correct | Complementary swap | Mismatched | No image |
|---|---|---|---|---|
| number | 2.251 | 3.381 | 3.085 | 1.987 |
| other | 2.246 | 3.384 | 3.581 | 3.286 |
| yes/no | 0.702 | 1.085 | 1.050 | 1.177 |
