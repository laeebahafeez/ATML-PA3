# Task 3 tables (held-out split unless stated)

Accuracy, oracle coverage and token cost are averaged over the 10 saved orders within a question, then over questions. Bootstrap intervals: 1,000 paired question-level resamples.


## MATH (Qwen2.5-3B): accuracy (%) by candidate budget

| Method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|---|
| Highest mean log-prob | 64.3 | 65.8 | 67.2 | 67.5 | 66.4 | 64.0 |
| Highest self-certainty | 64.3 | 66.8 | 68.8 | 70.1 | 70.7 | 70.6 |
| FirstAns | 64.3 | 64.3 | 64.3 | 64.3 | 64.3 | 64.3 |
| Self-consistency | 64.3 | 64.3 | 69.2 | 71.8 | 73.1 | 73.8 |
| Self-certainty + RWC | 64.3 | 66.8 | 69.7 | 72.9 | 74.6 | 75.6 |
| Shortest rollout | 64.3 | 65.3 | 65.2 | 64.6 | 63.2 | 62.4 |
| *Oracle coverage* | 64.3 | 72.5 | 79.1 | 84.3 | 88.5 | 91.4 |
| *Distinct answer groups* | 1.00 | 1.37 | 1.93 | 2.74 | 4.00 | 5.91 |
| *Tokens (all candidates)* | 670 | 1340 | 2680 | 5344 | 10678 | 21347 |

Paired differences in accuracy (pp, 95% CI):

| Comparison | N=8 | N=32 |
|---|---|---|
| Highest mean log-prob − FirstAns | +3.2 [+1.9, +4.4] | -0.4 [-2.5, +1.8] |
| Highest self-certainty − Highest mean log-prob | +2.6 [+1.3, +3.8] | +6.6 [+3.9, +9.1] |
| Highest self-certainty − Self-consistency | -1.7 [-2.8, -0.4] | -3.2 [-5.3, -0.8] |
| Self-consistency − FirstAns | +7.4 [+6.3, +8.6] | +9.4 [+7.9, +11.2] |
| Self-certainty + RWC − Highest self-certainty | +2.8 [+1.9, +3.8] | +4.9 [+2.8, +7.1] |
| Self-certainty + RWC − Self-consistency | +1.1 [+0.5, +1.8] | +1.8 [+0.7, +3.0] |

Failure decomposition at N=32 (% of question-orders):

| Method | Correct | Selection failure | Sampling failure |
|---|---|---|---|
| Highest mean log-prob | 64.0 | 27.4 | 8.6 |
| Highest self-certainty | 70.6 | 20.8 | 8.6 |
| FirstAns | 64.3 | 27.0 | 8.6 |
| Self-consistency | 73.8 | 17.6 | 8.6 |
| Self-certainty + RWC | 75.6 | 15.8 | 8.6 |
| Shortest rollout | 62.4 | 29.0 | 8.6 |

Signals at N=32 (within-question; 'mixed' = questions with some correct and some incorrect candidates):

| Signal | Top−bottom quartile correct rate (all) | (mixed) | Spearman with correctness (mixed) | Spearman with length (mixed) |
|---|---|---|---|---|
| logp | +7.8 | +13.4 | +0.134 | +0.245 |
| selfcert | +12.0 | +20.4 | +0.198 | -0.193 |
| length | -4.6 | -7.8 | -0.076 | +1.000 |

Pool: mean candidate accuracy 64.5%, all-correct 32.8%, none-correct 8.6%, mean length correct 534 vs incorrect 909 tokens, capped 1.7%, NO_ANSWER 1.7%.

Adaptive stopping (quality signal: selfcert); settings chosen on the dev split:

| Policy | τ | γ | Dev acc | Held-out acc | Δ vs fixed-32 (pp, 95% CI) | Mean tokens | Mean rollouts | Stops at 4 / 8 / 16 / 32 (%) | Wrong early stops (fixable by fixed-32) |
|---|---|---|---|---|---|---|---|---|---|
| agreement | 0.625 | — | 82.6 | 73.6 | -0.15 [-0.50, +0.16] | 10140 | 11.82 | 66.3 / 5.9 / 1.3 / 26.5 | 660 (51) |
| quality | 0.5 | 0.0 | 82.2 | 73.2 | -0.60 [-1.20, -0.01] | 7658 | 9.00 | 76.4 / 5.9 / 1.3 / 16.4 | 1153 (131) |
| fixed32 | — | — | 82.5 | 73.8 | — | 21347 | 32.00 | 0.0 / 0.0 / 0.0 / 100.0 | 0 (0) |

Fixed-N self-consistency at comparable cost: N=4: 69.2% at 2680 tokens, N=8: 71.8% at 5344 tokens, N=16: 73.1% at 10678 tokens.

Stopping by full-pool composition (retrospective; question-orders):

| Policy | Grouped by | Bin | Runs | Mean rollouts | Stop at 4 | Accuracy | Fixed-32 accuracy |
|---|---|---|---|---|---|---|---|
| agreement | fraction of correct candidates | 0 | 690 | 23.4 | 25.1 | 0.0 | 0.0 |
| agreement | fraction of correct candidates | (0, 0.5] | 2200 | 24.2 | 21.3 | 36.3 | 36.0 |
| agreement | fraction of correct candidates | (0.5, 1) | 2490 | 5.9 | 82.0 | 99.3 | 100.0 |
| agreement | fraction of correct candidates | 1 | 2620 | 4.0 | 100.0 | 100.0 | 100.0 |
| agreement | largest answer-group share | ≤ 0.5 | 2380 | 27.0 | 13.6 | 33.2 | 33.3 |
| agreement | largest answer-group share | (0.5, 0.75] | 1070 | 10.6 | 53.7 | 68.5 | 69.2 |
| agreement | largest answer-group share | (0.75, 1] | 4550 | 4.2 | 96.8 | 96.0 | 96.0 |
| quality | fraction of correct candidates | 0 | 690 | 18.5 | 39.7 | 0.0 | 0.0 |
| quality | fraction of correct candidates | (0, 0.5] | 2200 | 16.9 | 44.2 | 35.4 | 36.0 |
| quality | fraction of correct candidates | (0.5, 1) | 2490 | 4.6 | 90.1 | 98.6 | 100.0 |
| quality | fraction of correct candidates | 1 | 2620 | 4.0 | 100.0 | 100.0 | 100.0 |
| quality | largest answer-group share | ≤ 0.5 | 2380 | 19.5 | 36.6 | 32.1 | 33.3 |
| quality | largest answer-group share | (0.5, 0.75] | 1070 | 6.6 | 73.0 | 67.9 | 69.2 |
| quality | largest answer-group share | (0.75, 1] | 4550 | 4.1 | 98.0 | 95.9 | 96.0 |

## MathVista (Qwen2.5-VL-3B): accuracy (%) by candidate budget

| Method | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|---|
| Highest mean log-prob | 54.5 | 55.4 | 54.5 | 53.1 | 51.7 | 49.4 |
| Highest self-certainty | 54.5 | 55.8 | 56.1 | 55.9 | 56.2 | 55.9 |
| Highest VisConf | 54.5 | 56.3 | 57.2 | 57.2 | 57.6 | 58.4 |
| Highest token-fused VisConf | 54.5 | 56.3 | 56.8 | 56.6 | 56.6 | 57.1 |
| Highest vision | 54.5 | 55.3 | 55.3 | 55.2 | 54.8 | 54.5 |
| FirstAns | 54.5 | 54.5 | 54.5 | 54.5 | 54.5 | 54.5 |
| Self-consistency | 54.5 | 54.5 | 59.0 | 61.4 | 62.8 | 64.1 |
| Self-certainty + RWC | 54.5 | 55.8 | 58.0 | 61.6 | 63.0 | 64.8 |
| Shortest rollout | 54.5 | 55.8 | 55.8 | 55.2 | 54.8 | 54.4 |
| VisConf + RWC | 54.5 | 56.3 | 58.8 | 62.1 | 63.5 | 65.0 |
| Token-fused VisConf + RWC | 54.5 | 56.3 | 58.6 | 62.3 | 63.6 | 64.5 |
| Vision + RWC | 54.5 | 55.3 | 57.5 | 61.3 | 62.8 | 64.5 |
| *Oracle coverage* | 54.5 | 67.1 | 76.8 | 83.6 | 88.0 | 90.2 |
| *Distinct answer groups* | 1.00 | 1.44 | 2.01 | 2.74 | 3.65 | 4.76 |
| *Tokens (all candidates)* | 278 | 551 | 1112 | 2230 | 4462 | 8898 |

Paired differences in accuracy (pp, 95% CI):

| Comparison | N=8 | N=32 |
|---|---|---|
| Highest self-certainty − Self-consistency | -5.5 [-7.0, -4.0] | -8.2 [-11.1, -5.1] |
| Highest VisConf − Highest self-certainty | +1.3 [-0.3, +2.8] | +2.5 [-0.9, +5.6] |
| Highest VisConf − Highest token-fused VisConf | +0.6 [-0.1, +1.3] | +1.2 [-0.6, +3.2] |
| Self-consistency − FirstAns | +6.8 [+5.6, +8.0] | +9.6 [+7.5, +11.5] |
| Self-certainty + RWC − Self-consistency | +0.3 [-0.5, +1.0] | +0.7 [-0.7, +2.2] |
| VisConf + RWC − Highest VisConf | +5.0 [+3.8, +6.1] | +6.6 [+3.9, +9.4] |
| VisConf + RWC − Self-consistency | +0.8 [-0.0, +1.7] | +0.9 [-0.4, +2.4] |
| VisConf + RWC − Token-fused VisConf + RWC | -0.2 [-0.5, +0.2] | +0.5 [-0.1, +1.1] |
| Vision + RWC − Self-consistency | -0.0 [-0.9, +0.9] | +0.4 [-1.0, +1.9] |

Failure decomposition at N=32 (% of question-orders):

| Method | Correct | Selection failure | Sampling failure |
|---|---|---|---|
| Highest mean log-prob | 49.4 | 40.9 | 9.8 |
| Highest self-certainty | 55.9 | 34.4 | 9.8 |
| Highest VisConf | 58.4 | 31.9 | 9.8 |
| Highest token-fused VisConf | 57.1 | 33.1 | 9.8 |
| Highest vision | 54.5 | 35.8 | 9.8 |
| FirstAns | 54.5 | 35.7 | 9.8 |
| Self-consistency | 64.1 | 26.2 | 9.8 |
| Self-certainty + RWC | 64.8 | 25.5 | 9.8 |
| Shortest rollout | 54.4 | 35.9 | 9.8 |
| VisConf + RWC | 65.0 | 25.2 | 9.8 |
| Token-fused VisConf + RWC | 64.5 | 25.8 | 9.8 |
| Vision + RWC | 64.5 | 25.8 | 9.8 |

Signals at N=32 (within-question; 'mixed' = questions with some correct and some incorrect candidates):

| Signal | Top−bottom quartile correct rate (all) | (mixed) | Spearman with correctness (mixed) | Spearman with length (mixed) |
|---|---|---|---|---|
| logp | +2.1 | +2.6 | +0.034 | +0.356 |
| selfcert | +3.7 | +4.6 | +0.050 | +0.351 |
| length | -3.6 | -4.5 | -0.044 | +1.000 |
| vision | +3.5 | +4.4 | +0.037 | -0.688 |
| visconf | +7.3 | +9.1 | +0.087 | -0.290 |
| visconf_token | +7.2 | +9.0 | +0.079 | -0.330 |

Pool: mean candidate accuracy 54.7%, all-correct 10.4%, none-correct 9.8%, mean length correct 220 vs incorrect 348 tokens, capped 1.5%, NO_ANSWER 9.7%.

Confidence–vision correlation within rollouts: mean over questions -0.211 (median -0.212, IQR -0.306 to -0.094; 10.6% of questions positive). Mean Z_token − Z = -0.00224 (= mean within-rollout covariance; identity error 5.1e-17).

Adaptive stopping (quality signal: visconf); settings chosen on the dev split:

| Policy | τ | γ | Dev acc | Held-out acc | Δ vs fixed-32 (pp, 95% CI) | Mean tokens | Mean rollouts | Stops at 4 / 8 / 16 / 32 (%) | Wrong early stops (fixable by fixed-32) |
|---|---|---|---|---|---|---|---|---|---|
| agreement | 0.5 | — | 63.6 | 63.1 | -0.95 [-2.03, +0.08] | 3029 | 8.86 | 75.0 / 7.5 / 2.0 / 15.4 | 1930 (333) |
| quality | 0.5 | 0.0 | 63.9 | 63.6 | -0.50 [-1.46, +0.38] | 3384 | 10.04 | 68.3 / 9.8 / 3.0 / 18.9 | 1709 (262) |
| fixed32 | — | — | 63.7 | 64.1 | — | 8898 | 32.00 | 0.0 / 0.0 / 0.0 / 100.0 | 0 (0) |

Fixed-N self-consistency at comparable cost: N=4: 59.0% at 1112 tokens, N=8: 61.4% at 2230 tokens, N=16: 62.8% at 4462 tokens.

Stopping by full-pool composition (retrospective; question-orders):

| Policy | Grouped by | Bin | Runs | Mean rollouts | Stop at 4 | Accuracy | Fixed-32 accuracy |
|---|---|---|---|---|---|---|---|
| agreement | fraction of correct candidates | 0 | 780 | 18.7 | 43.7 | 0.0 | 0.0 |
| agreement | fraction of correct candidates | (0, 0.5] | 2900 | 12.6 | 59.1 | 29.9 | 27.8 |
| agreement | fraction of correct candidates | (0.5, 1) | 3490 | 4.7 | 89.3 | 96.1 | 100.0 |
| agreement | fraction of correct candidates | 1 | 830 | 4.0 | 100.0 | 100.0 | 100.0 |
| agreement | largest answer-group share | ≤ 0.5 | 2450 | 15.0 | 52.7 | 31.1 | 32.9 |
| agreement | largest answer-group share | (0.5, 0.75] | 2190 | 8.0 | 70.8 | 58.2 | 59.8 |
| agreement | largest answer-group share | (0.75, 1] | 3360 | 5.0 | 94.1 | 89.7 | 89.6 |
| quality | fraction of correct candidates | 0 | 780 | 20.2 | 36.3 | 0.0 | 0.0 |
| quality | fraction of correct candidates | (0, 0.5] | 2900 | 14.8 | 48.3 | 30.0 | 27.8 |
| quality | fraction of correct candidates | (0.5, 1) | 3490 | 5.2 | 84.5 | 97.0 | 100.0 |
| quality | fraction of correct candidates | 1 | 830 | 4.0 | 100.0 | 100.0 | 100.0 |
| quality | largest answer-group share | ≤ 0.5 | 2450 | 17.7 | 40.0 | 31.8 | 32.9 |
| quality | largest answer-group share | (0.5, 0.75] | 2190 | 9.0 | 63.6 | 59.1 | 59.8 |
| quality | largest answer-group share | (0.75, 1] | 3360 | 5.1 | 92.0 | 89.7 | 89.6 |

## Live MATH strategies (50 held-out questions, 3,072-token output allowance)

| Strategy | Accuracy | Mean input tokens | Mean output tokens | Mean seconds | Truncated (any call) | Answer extracted |
|---|---|---|---|---|---|---|
| parallel | 54.0 | 325 | 1857 | 100.4 | 26.0 | 78.0 |
| refine | 62.0 | 890 | 1217 | 66.3 | 20.0 | 100.0 |
| single | 62.0 | 108 | 760 | 41.5 | 4.0 | 96.0 |

Per call: parallel: truncated 16.7%, extracted 83.3%, 619 output tokens; refine: truncated 0.0%, extracted 100.0%, 582 output tokens; single: truncated 4.0%, extracted 96.0%, 760 output tokens.

Refinement transitions vs the shared initial answer: {'C->C': 28, 'W->W': 19, 'W->C': 3}; answer changed in 20.0%; initial accuracy 56.0%.

Paired accuracy differences (pp, 95% CI): parallel-single: -8 [-20, +6]; refine-single: +0 [-14, +12]; refine-parallel: +8 [-2, +18].

## Illustrative MathVista trajectories (figure mathvista_trajectories.png)

| Question | Rollout | Correct | Tokens | corr(c, v) | C | V | Z = V·C | Z_token |
|---|---|---|---|---|---|---|---|---|
| 99 | 0 | True | 108 | -0.15 | 0.446 | 0.058 | 0.0258 | 0.0240 |
| 99 | 7 | False | 87 | -0.43 | 0.445 | 0.050 | 0.0221 | 0.0177 |
| 425 | 0 | True | 259 | -0.34 | 0.608 | 0.013 | 0.0076 | 0.0067 |
| 425 | 22 | False | 2048 | +0.05 | 0.307 | 0.005 | 0.0017 | 0.0017 |
