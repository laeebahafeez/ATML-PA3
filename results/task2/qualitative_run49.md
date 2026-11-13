# Complementary-image examples: run49

Selection rule: for each suite and question type, pairs in pair-index order; the first 1 pair(s) of each outcome category (grounded / image ignored / differ-but-wrong / right-but-swap-insensitive) are shown. References are the original answers; predictions are greedy, 16 tokens.


### Annotation-derived grounding / counting: pair 32 — grounded: right for both images, and the swap moves each answer to the partner's

Question: How many cars can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000338562.jpg" height="120"> | 1 | `1` | `2` | `1` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000542042.jpg" height="120"> | 2 | `2` | `1` | `1` | `1` |

### Annotation-derived grounding / counting: pair 33 — same answer for both images (image ignored)

Question: How many ties can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000369461.jpg" height="120"> | 1 | `1` | `1` | `1` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000440106.jpg" height="120"> | 2 | `1` | `1` | `1` | `1` |

### Annotation-derived grounding / counting: pair 34 — answers differ between the images, but at least one is wrong

Question: How many pairs of skis can you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000573686.jpg" height="120"> | 3 | `2` | `1` | `1` | `2` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000301581.jpg" height="120"> | 1 | `1` | `2` | `1` | `2` |

### Annotation-derived grounding / presence: pair 0 — grounded: right for both images, and the swap moves each answer to the partner's

Question: Does this image contain a cell phone?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000572210.jpg" height="120"> | yes | `yes` | `no` | `no` | `no` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000487861.jpg" height="120"> | no | `no` | `yes` | `no` | `no` |

### Annotation-derived grounding / presence: pair 2 — same answer for both images (image ignored)

Question: Does this image contain a vase?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000498716.jpg" height="120"> | yes | `yes` | `yes` | `no` | `yes` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000362094.jpg" height="120"> | no | `yes` | `yes` | `no` | `yes` |

### Natural VQAv2 / number: pair 36 — answers differ between the images, but at least one is wrong

Question: How many animals do you see?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000199003.jpg" height="120"> | 3 | `4` | `2` | `2` | `2` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000199849.jpg" height="120"> | 6 | `2` | `4` | `2` | `2` |

### Natural VQAv2 / number: pair 39 — same answer for both images (image ignored)

Question: How many clocks are showing?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000037389.jpg" height="120"> | 2 | `1` | `1` | `0` | `1` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000007297.jpg" height="120"> | 1 | `1` | `1` | `1` | `1` |

### Natural VQAv2 / number: pair 42 — grounded: right for both images, and the swap moves each answer to the partner's

Question: How many horses are shown?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000024260.jpg" height="120"> | 2 | `2` | `1` | `0` | `2` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000464633.jpg" height="120"> | 1 | `1` | `2` | `0` | `2` |

### Natural VQAv2 / other: pair 60 — same answer for both images (image ignored)

Question: What color is her dress?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000459721.jpg" height="120"> | white | `white` | `white` | `black` | `white` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000068435.jpg" height="120"> | red | `white` | `white` | `red` | `white` |

### Natural VQAv2 / other: pair 65 — answers differ between the images, but at least one is wrong

Question: What city is this?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000475244.jpg" height="120"> | sydney | `southeast` | `new york` | `new york` | `New York` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000253413.jpg" height="120"> | london | `new york` | `southeast` | `new york` | `New York` |

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

### Natural VQAv2 / yes/no: pair 1 — grounded: right for both images, and the swap moves each answer to the partner's

Question: Is the picture taken indoors?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000048376.jpg" height="120"> | no | `no` | `yes` | `yes` | `yes` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000391090.jpg" height="120"> | yes | `yes` | `no` | `yes` | `yes` |

### Natural VQAv2 / yes/no: pair 19 — answers differ between the images, but at least one is wrong

Question: Are posters on the wall?

| | Image | Reference | Correct | Swap (partner image) | Mismatched | No image |
|---|---|---|---|---|---|---|
| image 1 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000008775.jpg" height="120"> | yes | `no` | `yes` | `no` | `yes` |
| image 2 | <img src="http://images.cocodataset.org/val2014/COCO_val2014_000000090839.jpg" height="120"> | no | `yes` | `no` | `no` | `yes` |
