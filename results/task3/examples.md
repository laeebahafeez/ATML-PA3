# Task 3 examples (rule-selected)

Rules: held-out questions in split order, saved order 0. *Disagreement*: the first question where self-consistency and the RWC method differ at N=32 with SC wrong and RWC right, and the first with the reverse. *Premature stop*: the first question where the quality-aware controller stopped early with a wrong answer that fixed-32 self-consistency got right. *Revision*: the first live question of each transition type, plus the first wrong-to-wrong case whose answer changed. ✓ marks the group graded correct (retrospective only).


## MATH (Qwen2.5-3B)


### Selection disagreement (self-consistency wrong, RWC right): test:2460

> At a competition with $N$ players, the number of players given elite status is equal to $2^{1+\lfloor \log_{2} (N-1) \rfloor}-N$. Suppose that $19$ players are given elite status. What is the sum of the two smallest possible values of $N$?

Gold `154`. Self-consistency → `58`; Self-certainty + RWC → `154`. Answer groups among the 32: `154` ×11 ✓, `58` ×11, `NO_ANSWER` ×3, `45` ×1, `98` ×1.

### Selection disagreement (self-consistency right, RWC wrong): test:2198

> Let $g(x) = x^2 - 11x + 30,$ and let $f(x)$ be a polynomial such that
\[g(f(x)) = x^4 - 14x^3 + 62x^2 - 91x + 42.\]Find the sum of all possible values of $f(10^{100}).$

Gold `11`. Self-consistency → `11`; Self-certainty + RWC → `14`. Answer groups among the 32: `11` ×5 ✓, `14` ×4, `10^{200} - 7 \cdot 10^{100} + 12` ×3, `10^{200} - 7 \cdot 10^{100} + 7` ×2, `30` ×2.

### Premature stop: test:1803 (stopped after 8 candidates)

> In right triangle $XYZ$, shown below, what is $\sin{X}$?

[asy]
draw((0,0)--(10,0)--(3.6,4.8)--cycle,black+linewidth(1));
draw(rightanglemark((0,0),(3.6,4.8),(10,0),20),black+linewidth(1));
label("X",(10,0),E);
label("Y",(3.6,4.8),N);
label("Z",(0,0),W);
label("10",(0,0)--(10,0),S);
label("8",(10,0)--(3.6,4.8),NE);
[/asy]

Gold `\frac35`. Revealed groups at the stop: `\frac{4}{5}` ×6, `\frac{3}{5}` ×2 ✓. All 32: `\frac{3}{5}` ×16 ✓, `\frac{4}{5}` ×15, `\frac{5}{4}` ×1.

## MathVista (Qwen2.5-VL-3B)


### Selection disagreement (self-consistency wrong, RWC right): 643

> As shown in the figure, CD is the diameter of ⊙O, chord DE ∥ OA, if the degree of ∠D is 50.0, then the degree of ∠C is ()
A. 25°
B. 30°
C. 40°
D. 50°

Gold `A`. Self-consistency → `C`; VisConf + RWC → `A`. Answer groups among the 32: `C` ×11, `A` ×10 ✓, `NO_ANSWER` ×8, `D` ×3.

### Selection disagreement (self-consistency right, RWC wrong): 552

> How many queries have a p-value lower than 0.50?

Gold `4`. Self-consistency → `4`; VisConf + RWC → `2`. Answer groups among the 32: `4` ×14 ✓, `3` ×9, `2` ×9.

### Premature stop: 434 (stopped after 4 candidates)

> In the figure above, side AC of triangle ABC is on line l. What is x in terms of k?
A. 60-k
B. k
C. 60+k
D. 120-k
E. 120-2*k

Gold `A`. Revealed groups at the stop: `C` ×2, `A` ×1 ✓, `NO_ANSWER` ×1. All 32: `A` ×17 ✓, `C` ×7, `B` ×3, `NO_ANSWER` ×2, `D` ×1.

## Live sequential refinement (MATH)


### W->C: test:196

> A circle of radius 5 with its center at $(0,0)$ is drawn on a Cartesian coordinate system. How many lattice points (points with integer coordinates) lie within or on this circle?

Initial answer `nan` → revised `81` (gold `81`).

End of initial response:

```
 (-3, 3), (-3, 4)\).

5. For \(x = \pm 4\):
   \[
   16 + y^2 \leq 25 \implies y^2 \leq 9 \implies -3 \leq y \leq 3
   \]
   This gives us 7 lattice points for each \(x = 4\) and \(x = -4\): \((4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3)\) and \((-4, -3), (-4, -2), (-4, -1), (-4, 0), (-
```

End of revision:

```
oints each (2 \times 9 = 18 points)
- For \(x = \pm 4\): 7 points each (2 \times 7 = 14 points)
- For \(x = \pm 5\): 1 point each (2 \times 1 = 2 points)

Adding these together:
\[
11 + 18 + 18 + 18 + 14 + 2 = 81
\]

Thus, the total number of lattice points within or on the circle is \(\boxed{81}\).
```

### W->W, answer changed: test:2177

> Let $(a_1,b_1),$ $(a_2,b_2),$ $\dots,$ $(a_n,b_n)$ be all the ordered pairs $(a,b)$ of complex numbers with $a^2+b^2\neq 0,$
\[a+\frac{10b}{a^2+b^2}=5, \quad \text{and} \quad b+\frac{10a}{a^2+b^2}=4.\]Find $a_1 + b_1 + a_2 + b_2 + \dots + a_n + b_n.$

Initial answer `nan` → revised `6` (gold `18`).

End of initial response:

```

\]
Let's try \( a = 2 \) and \( b = -2 \):
\[
2 + \frac{10 \cdot (-2)}{a^2 + b^2} = 5 \implies 2 - \frac{20}{4 + 4} = 5 \implies 2 - \frac{20}{8} = 5 \implies 2 - 2.5 = 5 \implies -0.5 = 5 \quad \text{(False)}
\]
Finally, let's try \( a = 2 \) and \( b = 1 \):
\[
2 + \frac{10 \cdot 1}{a^2 + b^2} = 
```

End of revision:

```
5 \implies 4 = 5 \quad \text{(False)}
\]
Finally, we find that the correct solution is \( a = 2 \) and \( b = 1 \) or \( a = 1 \) and \( b = 2 \). Therefore, the sum \( a + b \) is:
\[
2 + 1 = 3 \quad \text{or} \quad 1 + 2 = 3
\]
Thus, the sum is:
\[
3 + 3 = 6
\]
The final answer is:
\[
\boxed{6}
\]
```
