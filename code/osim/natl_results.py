"""Summaries and figures of the North Atlantic simulation for reports/glorys_chain.qmd."""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from osim import natl_sim as ns

LAYERS = (*ns.LAYER_NAMES, "total")
LAYER_LAB = {"ohc_15_700": "15-700 m", "ohc_700_2000": "700-2000 m", "total": "15-2000 m"}
DESIGN_LAB = {"lagrangian": "advected floats", "stationary": "stationary floats", "uniform": "iid uniform"}
EST_LAB = {"naive": "sample mean", "hajek": "Hajek", "mean_field": "mean field", "krige": "map", "dr": "DR"}
DIAG = {"": "qhat_diag", "wide": "qhat_diag_wide", "leaky": "qhat_diag_v1_leaky"}


def design_errors():
    """Per replicate, year and layer: the sample-mean error (GJ/m2) of the three sampling designs."""
    u = pd.read_csv(os.path.join(ns.OUT, "uniform_errors.csv")).assign(err=lambda x: x.est - x.theta)
    d = {"lagrangian": pd.read_csv(os.path.join(ns.OUT, "naive_errors.csv")),
         "stationary": pd.read_csv(os.path.join(ns.OUT, "stationary_errors.csv")), "uniform": u}
    return pd.concat([x[["rep", "year", "layer", "n", "err"]].assign(group=k) for k, x in d.items()], ignore_index=True)


def quadrature_remainder():
    """Q_y(Z) - theta_y (GJ/m2) per layer and year, with the layer sum as "total"."""
    q = pd.read_csv(os.path.join(ns.OUT, "quadrature.csv")).assign(rem=lambda x: x.remainder * 1e-9)
    q = pd.concat([q, q.groupby("year", as_index=False).rem.sum().assign(layer="total")])
    return q.set_index(["layer", "year"]).rem


def estimates(sig=ns.QSIG):
    """Per replicate, year, layer and estimator: error (GJ/m2), and DR's jackknife variance and df."""
    tag = os.path.basename(ns.q_dir(sig))
    e = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ns.OUT, "maps", "r*", f"estimates_{tag}.parquet"))])
    x = e.melt(id_vars=["rep", "layer", "year", "theta", "dr_V", "dr_df", "n"], value_vars=list(EST_LAB),
               var_name="group", value_name="est")
    return x.assign(err=(x.est - x.theta) * 1e-9, se=np.sqrt(x.dr_V) * 1e-9)


def rep_summary(E, by=("group", "layer")):
    """Over replicates, of each replicate's 2015-2025 mean error: bias, its Monte Carlo se, SD and RMSE; plus the
    mean over years of the single-year RMSE."""
    by = list(by)
    m = E.groupby([*by, "rep"]).err.mean()
    g = m.groupby(by)
    yr = E.groupby([*by, "year"]).err.apply(lambda x: np.sqrt((x ** 2).mean())).groupby(by).mean()
    return pd.DataFrame(dict(R=g.size(), bias=g.mean(), mc_se=g.std() / np.sqrt(g.size()), sd=g.std(),
                             rmse=np.sqrt((m ** 2).groupby(by).mean()), rmse_year=yr))


def paired(E, a, b, by=("layer",)):
    """Mean and Monte Carlo se over replicates of the 2015-2025 mean error difference a - b (same replicates)."""
    w = E[E.group.isin([a, b])].groupby(["group", *by, "rep"]).err.mean().unstack("group").dropna()
    d = (w[a] - w[b]).groupby(list(by))
    return pd.DataFrame(dict(diff=d.mean(), mc_se=d.std() / np.sqrt(d.size()), corr=w.groupby(list(by)).apply(
        lambda x: x[a].corr(x[b]))))


def year_corr(E, lags=(1, 5)):
    """Mean over year pairs of the across-replicate correlation of errors `lag` years apart, per group and layer."""
    def f(g):
        w = g.pivot(index="rep", columns="year", values="err")
        return pd.Series({f"lag{k}": np.mean([w[y].corr(w[y + k]) for y in w.columns if y + k in w.columns])
                          for k in lags})
    return E.groupby(["group", "layer"])[["rep", "year", "err"]].apply(f)


def plot_by_year(E, labels, layer, rem=None):
    """Bias (+/- 2 Monte Carlo se) and SD over replicates by year, one line per group; `rem` adds Q_y(Z) - theta_y."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for k, lab in labels.items():
        g = E[(E.group == k) & (E.layer == layer)].groupby("year").err
        b, s, n = g.mean(), g.std(), g.size()
        ax[0].errorbar(b.index, b, 2 * s / np.sqrt(n), marker="o", ms=3, capsize=2, label=lab)
        ax[1].plot(s.index, s, marker="o", ms=3, label=lab)
    if rem is not None:
        r = rem.loc[layer]
        ax[0].plot(r.index, r, "k--", lw=1, label="quadrature remainder")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(ylabel="bias (GJ/m2)", title="bias +/- 2 MC se")
    ax[1].set(ylabel="SD (GJ/m2)", title="SD over replicates", ylim=(0, None))
    ax[0].legend(fontsize=8)
    return fig


def qhat_metrics(kind=""):
    """qhat_diag metrics averaged over the diagnostic years."""
    m = pd.read_csv(os.path.join(ns.OUT, DIAG[kind], "metrics.csv"))
    return m.groupby(["sig_deg", "sig_days", "source"]).mean(numeric_only=True).drop(columns="year").reset_index()


def realized_naive(years):
    """Realized naive bias of the total and its Monte Carlo se over the given years (advected floats)."""
    a = pd.read_csv(os.path.join(ns.OUT, "naive_errors.csv"))
    m = a[(a.layer == "total") & a.year.isin(years)].groupby("rep").err.mean()
    return m.mean(), m.std() / np.sqrt(len(m))


def plot_implied_bias():
    """Implied naive bias int Z q-hat / int q-hat - theta against the spatial kernel sd, per temporal sd (daily
    positions), for the domain-restricted estimator and the first (ocean-support) version, with the realized bias."""
    m = pd.concat([qhat_metrics(""), qhat_metrics("wide")])
    lk = qhat_metrics("leaky")
    years = pd.read_csv(os.path.join(ns.OUT, DIAG[""], "metrics.csv")).year.unique()
    b, se = realized_naive(years)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    ax.axhspan(b - 2 * se, b + 2 * se, color="0.85", label="realized bias +/- 2 MC se")
    for i, st in enumerate(sorted(m.sig_days.unique())):
        for d, ls, lab in ((m, "-", "domain-restricted"), (lk, ":", "ocean support")):
            g = d[(d.source == "daily") & (d.sig_days == st)]
            if len(g):
                ax.plot(g.sig_deg, g.bias_naive, ls, color=f"C{i}", marker="o", ms=3, label=f"{lab}, {st:g} d")
    ax.set(xscale="log", xlabel="spatial kernel sd (deg)", ylabel="implied bias (GJ/m2)")
    ax.set_xticks([1 / 12, 1 / 6, 1 / 3, 1 / 2, 1, 2, 3, 5], ["1/12", "1/6", "1/3", "1/2", "1", "2", "3", "5"])
    ax.legend(fontsize=7, ncol=2)
    return fig


def qhat_table(sig_days=30.0):
    """Precision and positivity of the held-out q-hat at one temporal sd, daily positions vs profiles."""
    m = pd.concat([qhat_metrics(""), qhat_metrics("wide")])
    m = m[m.sig_days == sig_days].pivot(index="sig_deg", columns="source",
                                        values=["rel_se", "n_zero_heldout", "q_q01", "q_q99", "ess_frac"])
    m.columns = [f"{a}_{b}" for a, b in m.columns]
    return m.reset_index()


def crosshalf():
    """Cross-half naive and Hajek errors per replicate, year, layer, kernel sd and source (GJ/m2)."""
    c = pd.read_csv(os.path.join(ns.OUT, DIAG[""], "crosshalf.csv"))
    s = c.sigma.str.extract(r"s([\d.]+)_t([\d.]+)").astype(float)
    c = c.assign(sig_deg=s[0], sig_days=s[1])
    return pd.concat([c.assign(group=g, err=c[g] - c.theta) for g in ("naive", "hajek")], ignore_index=True)


def plot_hajek_bandwidth(C, layer, source="profiles"):
    """Cross-half Hajek bias (+/- 2 MC se) and RMSE of the 3-year mean against the spatial kernel sd, per temporal
    sd; the sample mean for reference."""
    C = C[(C.layer == layer) & (C.source == source)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    nv = rep_summary(C[C.group == "naive"].assign(group="naive"), ["group"]).iloc[0]
    for a, v in zip(ax, (nv.bias, nv.rmse)):
        a.axhline(v, color="k", ls="--", lw=1, label="sample mean")
    for i, st in enumerate((10.0, 30.0)):
        t = rep_summary(C[(C.group == "hajek") & (C.sig_days == st)], ["sig_deg"])
        ax[0].errorbar(t.index, t.bias, 2 * t.mc_se, marker="o", ms=3, capsize=2, color=f"C{i}", label=f"Hajek, {st:g} d")
        ax[1].plot(t.index, t.rmse, marker="o", ms=3, color=f"C{i}", label=f"Hajek, {st:g} d")
    ax[0].axhline(0, color="k", lw=0.6)
    for a in ax:
        a.set(xscale="log", xlabel="spatial kernel sd (deg)")
        a.set_xticks([1 / 12, 1 / 6, 1 / 3, 1 / 2, 1, 2, 3, 5], ["1/12", "1/6", "1/3", "1/2", "1", "2", "3", "5"])
    ax[0].set(ylabel="bias (GJ/m2)", title="bias +/- 2 MC se")
    ax[1].set(ylabel="RMSE (GJ/m2)", title="RMSE", ylim=(0, None))
    ax[0].legend(fontsize=8)
    return fig


def plot_qhat_maps(year=2020):
    """Annual-mean q-hat / q-bar (daily positions) at several kernel sds, on one log colour scale."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LogNorm
    la, lo, m, _ = ns.domain()
    panels = []
    for kind in ("", "wide"):
        z = np.load(os.path.join(ns.OUT, DIAG[kind], f"maps_{year}.npz"))
        for k in [k for k in z.files if not k.startswith(("lat", "lon"))]:
            s = k.split("_")[0][1:]
            panels.append((k, z[f"lon_s{s}"], z[f"lat_s{s}"], z[k]))
    fig, ax = plt.subplots(2, 3, figsize=(13.2, 7), squeeze=False, subplot_kw=dict(projection=ccrs.PlateCarree()),
                           layout="constrained")
    for a, (k, x, y, q) in zip(ax.flat, panels):
        s, t = k[1:].split("_t")
        pc = a.pcolormesh(x, y, np.where(q > 0, q, np.nan), shading="nearest", norm=LogNorm(0.5, 2), cmap="RdBu_r",
                          rasterized=True, transform=ccrs.PlateCarree())
        a.contour(lo, la, m.astype(float), levels=[0.5], colors="k", linewidths=0.5, transform=ccrs.PlateCarree())
        a.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        a.coastlines(lw=0.4, zorder=3)
        a.set_extent([lo.min(), lo.max(), ns.SOUTH, la.max()], crs=ccrs.PlateCarree())
        a.set_title(f"sd {float(s):.3g} deg, {t} d", fontsize=9)
    for a in ax.flat[len(panels):]:
        a.set_visible(False)
    fig.colorbar(pc, ax=ax, shrink=0.6).set_label(f"q-hat / q-bar, {year} mean")
    return fig


def coverage(E, level=0.95):
    """Coverage of DR's t interval (jackknife variance, Satterthwaite df) per layer, over replicate-years."""
    from scipy import stats
    d = E[E.group == "dr"]
    hit = np.abs(d.err) <= stats.t.ppf(0.5 + level / 2, d.dr_df) * d.se
    return pd.DataFrame(dict(coverage=hit.groupby(d.layer).mean(), median_se=d.se.groupby(d.layer).median(),
                             sd_err=d.groupby(["layer", "year"]).err.std().groupby("layer").median()))
