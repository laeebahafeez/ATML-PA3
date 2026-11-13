# Complementary-image examples: run196

Selection rule: for each suite and question type, pairs in pair-index order; the first 1 pair(s) of each outcome category (grounded / image ignored / differ-but-wrong / right-but-swap-insensitive) are shown. References are the original answers; predictions are greedy, 16 tokens.


### Annotation-derived grounding / counting: pair 32 — same answer for both images (image ignored)

Question: How many cars can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000338562.jpg" height="120"> | 1 | `2` | `2` | `0` | `2` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000542042.jpg" height="120"> | 2 | `2` | `2` | `1` | `2` |

### Annotation-derived grounding / counting: pair 33 — answers differ between the images, but at least one is wrong

Question: How many ties can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000369461.jpg" height="120"> | 1 | `2` | `1` | `1` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000440106.jpg" height="120"> | 2 | `1` | `2` | `1` | `1` |

### Annotation-derived grounding / counting: pair 36 — grounded: right for both images, and the swap moves each answer to the partner's

Question: How many baseball gloves can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000185240.jpg" height="120"> | 1 | `1` | `2` | `0` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000554328.jpg" height="120"> | 2 | `2` | `1` | `1` | `1` |

### Annotation-derived grounding / presence: pair 0 — grounded: right for both images, and the swap moves each answer to the partner's

Question: Does this image contain a cell phone?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000572210.jpg" height="120"> | yes | `yes` | `no` | `no` | `no` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000487861.jpg" height="120"> | no | `no` | `yes` | `no` | `no` |

### Annotation-derived grounding / presence: pair 5 — same answer for both images (image ignored)

Question: Does this image contain a pizza?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000528821.jpg" height="120"> | yes | `yes` | `yes` | `no` | `no` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000310354.jpg" height="120"> | no | `yes` | `yes` | `no` | `no` |

### Natural VQAv2 / number: pair 36 — answers differ between the images, but at least one is wrong

Question: How many animals do you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000199003.jpg" height="120"> | 3 | `3` | `2` | `2` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000199849.jpg" height="120"> | 6 | `2` | `3` | `2` | `1` |

### Natural VQAv2 / number: pair 39 — same answer for both images (image ignored)

Question: How many clocks are showing?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000037389.jpg" height="120"> | 2 | `1` | `1` | `1` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000007297.jpg" height="120"> | 1 | `1` | `1` | `1` | `1` |

### Natural VQAv2 / number: pair 42 — grounded: right for both images, and the swap moves each answer to the partner's

Question: How many horses are shown?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000024260.jpg" height="120"> | 2 | `2` | `1` | `1` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000464633.jpg" height="120"> | 1 | `1` | `2` | `0` | `1` |

### Natural VQAv2 / other: pair 60 — same answer for both images (image ignored)

Question: What color is her dress?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000459721.jpg" height="120"> | white | `white` | `white` | `white` | `blue` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000068435.jpg" height="120"> | red | `white` | `white` | `white` | `blue` |

### Natural VQAv2 / other: pair 62 — answers differ between the images, but at least one is wrong

Question: What color is the boy's hair?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000080172.jpg" height="120"> | blonde | `brown` | `black` | `black` | `brown` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000247407.jpg" height="120"> | brown | `black` | `brown` | `black` | `brown` |

### Natural VQAv2 / other: pair 66 — grounded: right for both images, and the swap moves each answer to the partner's

Question: What is the gender of the primary individual in the picture?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000422383.jpg" height="120"> | male | `male` | `female` | `male` | `female` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000199602.jpg" height="120"> | female | `female` | `male` | `male` | `female` |

### Natural VQAv2 / yes/no: pair 0 — same answer for both images (image ignored)

Question: Are these kitchen utensils?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000546826.jpg" height="120"> | no | `yes` | `yes` | `no` | `yes` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000392850.jpg" height="120"> | yes | `yes` | `yes` | `no` | `yes` |

### Natural VQAv2 / yes/no: pair 1 — answers differ between the images, but at least one is wrong

Question: Is the picture taken indoors?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000048376.jpg" height="120"> | no | `yes` | `no` | `yes` | `yes` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000391090.jpg" height="120"> | yes | `no` | `yes` | `no` | `yes` |

### Natural VQAv2 / yes/no: pair 4 — grounded: right for both images, and the swap moves each answer to the partner's

Question: Is anyone sitting on the bench?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000018485.jpg" height="120"> | no | `no` | `yes` | `no` | `no` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000017095.jpg" height="120"> | yes | `yes` | `no` | `no` | `no` |
