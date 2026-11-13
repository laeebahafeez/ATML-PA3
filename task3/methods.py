"""Selection / aggregation rules over a revealed candidate prefix.

Every function receives `q` (question dict from task3.data) and `idx`, the rollout ids
revealed so far in the current order (earliest first), and returns the index *into idx*
of a representative rollout whose answer group is the method's answer. Ties are broken
by the earliest supporting rollout in the current order.
"""
import numpy as np
from scipy.stats import rankdata


def first_answer(q, idx):
    return 0


def _vote(gids, weights):
    """Group with largest total weight; ties -> group whose earliest member comes first."""
    tot = {}
    first = {}
    for pos, (g, w) in enumerate(zip(gids, weights)):
        tot[g] = tot.get(g, 0.0) + w
        first.setdefault(g, pos)
    best = max(tot, key=lambda g: (tot[g], -first[g]))
    return first[best], tot


def self_consistency(q, idx):
    pos, _ = _vote(q["gid"][idx], np.ones(len(idx)))
    return pos


def highest(score):
    """Pick max score; np.argmax returns the first (= earliest) maximum."""
    return int(np.argmax(score))


def rwc_weights(score):
    n = len(score)
    if n == 1:
        return np.ones(1)
    r = rankdata(-np.asarray(score), method="average")     # rank 1 = best, ties averaged
    return (n - r) / (n - 1)


def rwc(q, idx, score):
    pos, _ = _vote(q["gid"][idx], rwc_weights(score))
    return pos


def rwc_margin(q, idx, score):
    """(top - runner-up) / total vote weight; 0 with a single group."""
    _, tot = _vote(q["gid"][idx], rwc_weights(score))
    vals = sorted(tot.values(), reverse=True)
    s = sum(vals)
    if len(vals) < 2 or s <= 0:
        return 0.0
    return (vals[0] - vals[1]) / s


# ---------------------------------------------------------------- quality scores
def score_L(q, idx):
    return q["L"][idx]


def score_K(q, idx):
    return q["K"][idx]


def score_V(q, idx):
    return q["V"][idx]


def calibrated_conf(q, idx):
    """Per-rollout token arrays c_{i,t}, calibrated on the pooled KL of the revealed set."""
    pooled = np.concatenate([q["kl_tokens"][i] for i in idx])
    if pooled.size == 0:
        return [np.zeros(0) for _ in idx]
    lo, hi = np.percentile(pooled, [5, 95])          # linear interpolation (numpy default)
    den = max(hi - lo, 1e-8)
    return [np.clip((q["kl_tokens"][i] - lo) / den, 0.0, 1.0) for i in idx]


def score_visconf(q, idx):
    c = calibrated_conf(q, idx)
    C = np.array([x.mean() if len(x) else 0.0 for x in c])
    return q["V"][idx] * C


def score_visconf_token(q, idx):
    c = calibrated_conf(q, idx)
    return np.array([(ci * q["v_tokens"][i]).mean() if len(ci) else 0.0 for ci, i in zip(c, idx)])


def score_neg_len(q, idx):
    """Length-only control: shortest rollout wins."""
    return -q["T"][idx].astype(float)


SCORES = {"logp": score_L, "selfcert": score_K, "vision": score_V,
          "visconf": score_visconf, "visconf_token": score_visconf_token, "shortest": score_neg_len}


def methods_for(dataset):
    """name -> callable(q, idx) -> position in idx."""
    m = {
        "first_ans": first_answer,
        "self_consistency": self_consistency,
        "best_logp": lambda q, i: highest(score_L(q, i)),
        "best_selfcert": lambda q, i: highest(score_K(q, i)),
        "selfcert_rwc": lambda q, i: rwc(q, i, score_K(q, i)),
        "shortest": lambda q, i: highest(score_neg_len(q, i)),
    }
    if dataset == "mathvista":
        m.update({
            "best_vision": lambda q, i: highest(score_V(q, i)),
            "vision_rwc": lambda q, i: rwc(q, i, score_V(q, i)),
            "best_visconf": lambda q, i: highest(score_visconf(q, i)),
            "visconf_rwc": lambda q, i: rwc(q, i, score_visconf(q, i)),
            "best_visconf_token": lambda q, i: highest(score_visconf_token(q, i)),
            "visconf_token_rwc": lambda q, i: rwc(q, i, score_visconf_token(q, i)),
        })
    return m
