# Review: PGD / AutoPGD / DAG / FGSM attack implementations in `attack_Main_v4_modified.py`

Date: 2026-08-02
Scope: meticulous verification of the PGD-family attack code, algorithm correctness, and
selected hyperparameters in `attack_Main_v4_modified.py` (current `master`).

No code has been changed as a result of this review — findings only, for follow-up.

---

## 1. CRITICAL — Attack direction is inverted (gradient descent instead of ascent)

Affects: `fgsm_attack` (L183–214), `pgd_attack` (L220–292), `dag_attack` (L360–427),
`autopgd_attack` (L429–510). `cw_attack` is NOT affected (uses a different, correct
construction via Adam minimizing a C&W margin loss).

Pattern repeated in all four functions:

```python
loss = -loss_dict['loss_ce']   # negate loss_ce, comment says "for maximization"
loss.backward()                # x.grad = d(-loss_ce)/dx = -grad(loss_ce)
...
x = x.detach() + step * grad.sign()   # ADD the sign of the (already negated) gradient
```

`grad` here already equals `-∇(loss_ce)`. Adding its sign to `x` is algebraically:

```
x_new = x + step * sign(-∇(loss_ce)) = x - step * sign(∇(loss_ce))
```

That is a **gradient descent** step on `loss_ce` (minimizes it), not ascent. `loss_ce` is
DETR's standard cross-entropy against ground-truth labels (`models/detr.py:121`,
plain `F.cross_entropy`, no unusual sign convention). The stated intent everywhere in the
code (comments: "gradient ascent step", "direction of increasing loss", "maximize loss_ce
(untargeted)") is to *increase* classification loss to cause misdetection/misclassification.
The actual arithmetic does the opposite: it nudges the image toward *lower* loss against the
true labels, i.e. away from the adversarial direction.

**Practical implication:** any "attack success rate" / robustness numbers produced via these
four functions are not measuring what they were intended to measure. This is the single most
important finding in this review.

**Fix (either one works, don't need both):**
- Drop the negation: `loss = loss_dict['loss_ce']; loss.backward(); x = x + step*grad.sign()`, or
- Keep the negation but flip the update sign: `loss = -loss_dict['loss_ce']; loss.backward(); x = x - step*grad.sign()`

---

## 2. CRITICAL — `autopgd_attack` random-start clamp corrupts the normalized image (L471)

```python
delta = torch.empty_like(original_img).uniform_(-eps, eps).to(device)
x_adv = torch.clamp(original_img + delta, 0.0, 1.0).detach()   # <-- bug
```

`img_tensors.tensors` is ImageNet mean/std-normalized
(`datasets/coco.py:126`, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), so its value
range is roughly **[-2.12, 2.64]**, not `[0,1]`. Clamping to `[0,1]` at initialization snaps
many pixels (anything below 0 or above 1 in normalized space, which is a large fraction of a
typical normalized image) far outside the intended epsilon ball, before any eps-ball
projection is applied. The very first gradient in each restart is therefore evaluated at a
distorted, not-actually-nearby point.

This is the exact bug already identified and disabled in the prior version (v3's `pgd_attack`
carries the comment `# clamp(, 0, 1) on NORMALIZED IMAGE IS WRONG` on the equivalent line,
with the random-init block fully commented out because of it). It has resurfaced,
uncommented, in the new `autopgd_attack`.

Note: the *later*, in-loop clamp is correctly commented out (L493:
`# x_adv = torch.clamp(x_adv, 0.0, 1.0)`), so only the pre-loop initialization is broken —
inconsistent with the rest of the function.

**Why this matters more than most bugs here:** `autopgd` is the attack path your current run
scripts actually use — `attack_coco_run.sh` and `attack_kitti_run.sh` both set
`attack='autopgd'`, `pgd_eps=0.5`, `autopgd_restarts=5`. This bug is live in current/recent
experiments.

**Fix:** don't clamp to `[0,1]`; just use `x_adv = original_img + delta` (already within the
eps ball by construction, since `delta` is drawn from `Uniform(-eps, eps)`).

---

## 3. HIGH — classic `pgd` attack path is dead/broken code

- L242–243:
  ```python
  if random_start:
      raise ValueError
      # # Random initialization within epsilon ball
      # noise = torch.FloatTensor(original_img.shape).uniform_(-eps, eps).to(device)
      # perturbed_img = torch.clamp(original_img + noise, 0, 1)  ## clamp(, 0, 1) ... IS WRONG.
  ```
  `--random_start` CLI default is `'True'` (L157), so `--attack_type pgd` with default flags
  crashes immediately on this `raise ValueError`. The real init code is commented out below
  it (for the same `[0,1]`-clamp reason as finding #2).

- Call site (L691–696): `pgd_attack`'s step-size parameter `alpha` is fed from `args.epsilon`
  (L694), whose CLI default is `0` (L138). If `--attack_type pgd` is run without *also*
  explicitly passing `--epsilon`, every PGD step adds `0 * grad.sign()` — the image is never
  perturbed, silently producing a "0% attack effect" result that looks like model robustness
  but is actually a no-op. `--pgd_eps` is the real perturbation budget; `--epsilon` is
  overloaded as PGD's step size, which is a confusing/dangerous naming collision (it's also
  used, unrelated, as FGSM's actual epsilon).

**Status:** not used by the current run scripts (which call `attack_type=autopgd`), but this
code is reachable and will crash or silently no-op for anyone who runs `--attack_type pgd`.

---

## 4. MEDIUM — "AutoPGD" doesn't fully match Croce & Hein's APGD algorithm

`autopgd_attack` implements a simplified adaptive-step PGD: shrink `step_size` by `rho` if the
latest loss is below the mean of the last `window` iterations. It does not implement:
- momentum in the update rule,
- checkpoint-based (fraction-of-budget) step-size halving schedule,
- resetting to the best point found so far when the step size is reduced.

Not a bug — a legitimate simplification — but worth not overclaiming as "AutoPGD (Croce &
Hein 2020)" in the paper without qualifying it, since a reviewer familiar with the original
algorithm could flag the discrepancy. Consider renaming (e.g. "adaptive-step PGD") or bringing
the implementation closer to canonical APGD if the citation matters.

---

## 5. LOW — hyperparameter unit ambiguity: `pgd_eps=0.5`

Both run scripts (`attack_coco_run.sh`, `attack_kitti_run.sh`) set `pgd_eps_list=(0.5)`. This
value is applied directly as the L∞ radius in normalized (mean/std-scaled) space. Given
std ≈ 0.225, `eps=0.5` in normalized space corresponds to roughly `0.5 × 0.225 ≈ 0.11` in raw
pixel units (~28/255) — a large, visually obvious perturbation budget, versus the code's own
default (`pgd_eps` default = `10/255 ≈ 0.039`) and typical robustness-literature budgets
(commonly 8/255). Not necessarily wrong, but the paper should state precisely which space
"eps=0.5" is defined in, since readers will otherwise assume raw pixel units.

---

## 6. LOW — dead CLI arguments left over from the DAG rewrite

`--dag_targeted` and `--dag_target_class` (L167–170) are still declared in the argument
parser, but `dag_attack`'s call site now hardcodes `targeted=False` (L711) and never passes
`target_class`. Harmless (no effect on results) but confusing/misleading if someone tries to
use `--dag_targeted` expecting it to do something.

---

## Priority order for fixes

1. Sign-flip / attack-direction bug (finding 1) — affects `fgsm`, `pgd`, `dag`, `autopgd`.
2. `autopgd_attack` init clamp (finding 2) — affects the attack path actually used in current
   run scripts.
3. Dead/broken `pgd` path (finding 3) — only matters if `--attack_type pgd` is ever invoked
   directly.
4. Naming precision for "AutoPGD" (finding 4) and `pgd_eps` units (finding 5) — paper-writing
   concerns, not code bugs.
5. Remove dead `--dag_targeted`/`--dag_target_class` args (finding 6) — cosmetic cleanup.
