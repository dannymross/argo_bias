"""Float survival by age, fitted to right-censored lifetimes, and its per-step form s(a)."""

import numpy as np
from scipy import stats


def fit(life, kind="km"):
    """Survival function S(t) (t in days since launch)."""
    dur = np.maximum(life["duration"].to_numpy(float), 1.0)
    data = stats.CensoredData.right_censored(dur, ~life["event"].to_numpy(bool))
    if kind == "km":
        return stats.ecdf(data).sf.evaluate
    if kind == "weibull":
        c, _, scale = stats.weibull_min.fit(data, floc=0)
        return lambda t: stats.weibull_min.sf(t, c, 0, scale)
    if kind == "exponential":
        rate = life["event"].sum() / dur.sum()
        return lambda t: np.exp(-rate * np.asarray(t, float))
    raise ValueError(kind)


def step_survival(sf, step_days, max_age_days):
    """s[a] = P(alive at age (a+1) steps | alive at age a steps)."""
    S = sf(np.arange(0.0, max_age_days + step_days, step_days))
    return np.divide(S[1:], S[:-1], out=np.zeros(len(S) - 1), where=S[:-1] > 0)
