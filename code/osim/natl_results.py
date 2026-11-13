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
    """Q_y(Z) - theta_y (GJ/m2) per layer (and "total") and year."""
    q = pd.read_csv(os.path.join(ns.OUT, "quadrature.csv")).assign(rem=lambda x: x.remainder * 1e-9)
    return q.set_index(["layer", "year"]).rem


def estimates(sig=None):
    """Per replicate, year, layer and estimator: error (GJ/m2), and DR's jackknife variance and df; `sig` a kernel
    (deg, days) or None for the per-layer production kernels."""
    tag = ns.QTAG if sig is None else os.path.basename(ns.q_dir(sig))
    e = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(ns.OUT, "maps", "r*", f"estimates_{tag}.parquet"))])
    x = e.melt(id_vars=["rep", "layer", "year", "theta", "dr_V", "dr_df", "n"], value_vars=list(EST_LAB),
               var_name="group", value_name="est")
    return x.assign(err=(x.est - x.theta) * 1e-9, se=np.sqrt(x.dr_V) * 1e-9)


def dr_bandwidth(sigs, ref=ns.QSIG, group="dr"):
    """Per kernel sd and layer: bias and mean over years of the MSE over replicates, and the paired difference in
    yearly MSE from `ref` with its Monte Carlo se (replicates common to all sds)."""
    E = pd.concat([(lambda e: e[e.group == group])(estimates(g)).assign(sig_deg=g[0], sig_days=g[1]) for g in sigs])
    E = E[E.groupby("rep").sig_deg.transform("size") == E.groupby("rep").size().max()]
    se2 = E.assign(se2=E.err ** 2).groupby(["sig_deg", "sig_days", "layer", "rep"])[["err", "se2"]].mean()
    d = se2.se2 - se2.se2.xs((*ref,), level=["sig_deg", "sig_days"])
    g = se2.groupby(["sig_deg", "sig_days", "layer"])
    dg = d.groupby(["sig_deg", "sig_days", "layer"])
    return pd.DataFrame(dict(R=g.size(), bias=g.err.mean(), mse_year=g.se2.mean(), d_mse=dg.mean(),
                             d_mse_se=dg.std() / np.sqrt(dg.size())))


def bandwidth_table(sigs, group="dr"):
    """Per kernel sd and layer: bias of the 2015-2025 mean (+ MC se), mean over years of |yearly bias|, max over
    years of |yearly bias|, and yearly MSE."""
    E = pd.concat([(lambda e: e[e.group == group])(estimates(g)).assign(sig_deg=g[0], sig_days=g[1]) for g in sigs])
    k = ["sig_deg", "sig_days", "layer"]
    yb = E.groupby([*k, "year"]).err.mean().abs().groupby(k)
    m = E.groupby([*k, "rep"]).err.mean().groupby(k)
    return pd.DataFrame(dict(bias=m.mean(), mc_se=m.sem(), mean_abs_year_bias=yb.mean(), max_abs_year_bias=yb.max(),
                             mse_year=E.assign(e2=E.err ** 2).groupby(k).e2.mean()))


SWEEP = [(s, t) for s in (0.3333, 0.5, 1.0, 2.0, 3.0, 5.0) for t in (2.0, 5.0, 10.0, 30.0, 90.0)]


def plot_dr_bandwidth(T, layer):
    """DR's mean |yearly bias| and yearly MSE against the spatial kernel sd, per temporal sd (bandwidth_table);
    the production kernel of the layer circled."""
    t = T.xs(layer, level="layer").reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for i, (st, g) in enumerate(t.groupby("sig_days")):
        ax[0].plot(g.sig_deg, g.mean_abs_year_bias, "o-", ms=3, color=f"C{i}", label=f"{st:g} d")
        ax[1].plot(g.sig_deg, g.mse_year * 1e3, "o-", ms=3, color=f"C{i}")
    if layer in ns.QSIG_LAYER:
        c = t[(t.sig_deg == ns.QSIG_LAYER[layer][0]) & (t.sig_days == ns.QSIG_LAYER[layer][1])]
        for a, v in zip(ax, (c.mean_abs_year_bias, c.mse_year * 1e3)):
            a.plot(c.sig_deg, v, "o", ms=11, mfc="none", mec="k", label="chosen")
    for a in ax:
        a.set(xscale="log", xlabel="spatial kernel sd (deg)")
        a.set_xticks([1 / 3, 1 / 2, 1, 2, 3, 5], ["1/3", "1/2", "1", "2", "3", "5"])
        a.minorticks_off()
    ax[0].set(ylabel="mean |yearly bias| (GJ/m2)", ylim=(0, None), title="DR bias")
    ax[1].set(ylabel="yearly MSE (1e-3 (GJ/m2)^2)", title="DR MSE")
    ax[0].legend(fontsize=8, title="temporal sd", title_fontsize=8)
    return fig


def spatial_diag():
    """Per-cell diagnostics written by natl_sim.spatial_diag."""
    return pd.read_csv(os.path.join(ns.OUT, "spatial_diag.csv"))


def plot_cell_maps(d, panels, ncol=2, one_cbar=False):
    """1 deg cell values on the native grid, masked to the domain. panels: (column, title, vmax or None for a
    symmetric limit at the 98th percentile of |value|, colour-bar label); d holds lat, lon cell centres."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    la, lo, m, _ = ns.domain()
    ci, cj = np.floor(la) + 0.5, np.floor(lo) + 0.5
    nr_ = -(-len(panels) // ncol)
    fig, ax = plt.subplots(nr_, ncol, figsize=(6.6 * ncol, 3.6 * nr_), squeeze=False, layout="constrained",
                           subplot_kw=dict(projection=ccrs.PlateCarree()))
    for a, (col, title, vmax, lab) in zip(ax.flat, panels):
        g = d.pivot_table(index="lat", columns="lon", values=col).reindex(index=ci, columns=cj)
        v = np.where(m, g.values, np.nan)
        vmax = vmax or np.nanquantile(np.abs(v), 0.98)
        pc = a.pcolormesh(lo, la, v, shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True,
                          transform=ccrs.PlateCarree())
        a.add_feature(cfeature.LAND, facecolor="0.85", zorder=2)
        a.coastlines(lw=0.4, zorder=3)
        a.set_extent([lo.min(), lo.max(), ns.SOUTH, la.max()], crs=ccrs.PlateCarree())
        a.set_title(title, fontsize=9)
        if not one_cbar:
            fig.colorbar(pc, ax=a, shrink=0.8).set_label(lab, fontsize=8)
    for a in ax.flat[len(panels):]:
        a.set_visible(False)
    if one_cbar:
        fig.colorbar(pc, ax=ax, shrink=0.5).set_label(lab)
    return fig


def plot_spatial(d, layer, kind):
    """Per-cell diagnostic maps of one layer. kind "residual": mean held-out residual at the profiles and its parts
    (common scale); "dr": r_j - 1 and DR's correction per unit area at the production kernels and, if the layer's
    kernel differs from QSIG, the change in the correction from QSIG."""
    prod = ",".join(f"{ns.QSIG_LAYER[l][0]:g}/{ns.QSIG_LAYER[l][1]:g}" for l in ns.LAYER_NAMES)
    base = ",".join(f"{ns.QSIG[0]:g}/{ns.QSIG[1]:g}" for _ in ns.LAYER_NAMES)
    g = d[d.layer == layer]
    w = g[g.kernel == prod].set_index(["lat", "lon"])
    if kind == "residual":
        cols = [("e_mean", "residual Z - map"), ("emf_mean", "Z - mean field"), ("anom_mean", "kriged anomaly")]
        vmax = np.nanquantile(np.abs(w[[c for c, _ in cols]].to_numpy()), 0.98)
        return plot_cell_maps(w.reset_index(), [(c, t, vmax, "GJ/m2") for c, t in cols], ncol=3)
    w = w.assign(r1=w.r - 1, cdens=w.c_share / w.area_share)
    panels = [("r1", "r_j - 1 (weight share / area share - 1)", 0.5, ""),
              ("cdens", "DR correction per unit area", None, "GJ/m2")]
    if ns.QSIG_LAYER[layer] != ns.QSIG:
        b = g[g.kernel == base].set_index(["lat", "lon"])
        w["dc"] = (w.c_share - b.c_share.reindex(w.index)) / w.area_share
        panels.append(("dc", f"change in correction from {ns.QSIG[0]:g} deg, {ns.QSIG[1]:g} d", None, "GJ/m2"))
    return plot_cell_maps(w.reset_index(), panels, ncol=len(panels))


def spatial_diag_year():
    """Per-cell diagnostics by year (natl_sim.spatial_diag, by_year=True)."""
    return pd.read_csv(os.path.join(ns.OUT, "spatial_diag_year.csv"))


def plot_spatial_years(dy, layer, kind):
    """One map per year, common scale: kind "residual" the mean held-out residual at the profiles, "dr" DR's
    correction per unit area, both at the production kernels."""
    prod = ",".join(f"{ns.QSIG_LAYER[l][0]:g}/{ns.QSIG_LAYER[l][1]:g}" for l in ns.LAYER_NAMES)
    g = dy[(dy.layer == layer) & (dy.kernel == prod)]
    v = g.e_mean if kind == "residual" else g.c_share / g.area_share
    w = g.assign(v=v).pivot_table(index=["lat", "lon"], columns="year", values="v")
    vmax = np.nanquantile(np.abs(w.to_numpy()), 0.98)
    w.columns = [f"y{c}" for c in w.columns]
    lab = "residual Z - map (GJ/m2)" if kind == "residual" else "DR correction (GJ/m2)"
    return plot_cell_maps(w.reset_index(), [(c, c[1:], vmax, lab) for c in w.columns], ncol=4, one_cbar=True)


def spatial_summary(d):
    """Key sums of the per-cell diagnostics at the production kernels: area-weighted residual parts, the correction
    total and the share of it in the top 20% of cells, the profile-mean map-bias proxy, and for 700-2000 m the change
    in the correction from QSIG, its positive-cell fraction and the share in cells straddling the domain edge."""
    prod = ",".join(f"{ns.QSIG_LAYER[l][0]:g}/{ns.QSIG_LAYER[l][1]:g}" for l in ns.LAYER_NAMES)
    base = ",".join(f"{ns.QSIG[0]:g}/{ns.QSIG[1]:g}" for _ in ns.LAYER_NAMES)
    out = {}
    for l, g in d.groupby("layer"):
        w = g[g.kernel == prod]
        top = w.c_share.abs().sort_values(ascending=False).index[: len(w) // 5]
        out[l] = dict(emf=(w.area_share * w.emf_mean).sum(), anom=(w.area_share * w.anom_mean).sum(),
                      proxy=w.map_contrib.sum(), corr=w.c_share.sum(), top20=w.c_share[top].sum() / w.c_share.sum())
        b = g[g.kernel == base].set_index(["lat", "lon"]).c_share
        dc = w.set_index(["lat", "lon"]).c_share - b
        edge = w.set_index(["lat", "lon"]).area_share < 0.8 * w.area_share.quantile(0.9)
        out[l] |= dict(dc=dc.sum(), dc_pos=(dc > 0).mean(), edge_area=w.set_index(["lat", "lon"]).area_share[edge].sum(),
                       edge_dc=dc[edge].sum() / dc.sum() if dc.abs().sum() > 0 else np.nan)
    return pd.DataFrame(out)


def select_bandwidth(sigs, layer="total", ref=ns.QSIG, split=60):
    """Choose the kernel sd on each half of the replicates (rep < split vs rep >= split) by DR's yearly MSE or by its
    mean over years of |yearly bias|, and evaluate the choice and `ref` on the other half."""
    E = pd.concat([(lambda e: e[(e.group == "dr") & (e.layer == layer)])(estimates(g)).assign(sig=str(g)) for g in sigs])
    crit = {"mse": lambda x: (x.err ** 2).groupby(x.sig).mean(),
            "abs_year_bias": lambda x: x.groupby(["sig", "year"]).err.mean().abs().groupby("sig").mean()}
    rows = []
    for c, f in crit.items():
        for name, h in {"first": E.rep < split, "second": E.rep >= split}.items():
            pick, ho = f(E[h]).idxmin(), f(E[~h])
            rows.append(dict(criterion=c, chosen_on=name, choice=pick, heldout=ho[pick], heldout_ref=ho[str(ref)],
                             heldout_best=ho.min(), heldout_best_sig=ho.idxmin()))
    return pd.DataFrame(rows)


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


def plot_by_year(E, labels, layer, rem=None, split=(), figsize=(11, 5), stats=("bias", "sd")):
    """Per year over replicates: bias (+/- 2 Monte Carlo se), SD and/or RMSE, one column each; the split groups get
    their own top row, colours as in plot_errors; `rem` adds Q_y(Z) - theta_y to the bottom bias panel."""
    lab = {"bias": ("bias (GJ/m2)", "bias +/- 2 MC se"), "sd": ("SD (GJ/m2)", "SD over replicates"),
           "rmse": ("RMSE (GJ/m2)", "RMSE over replicates")}
    groups = [g for g in ({k: v for k, v in labels.items() if k in split},
                          {k: v for k, v in labels.items() if k not in split}) if g]
    fig, ax = plt.subplots(len(groups), len(stats), figsize=figsize, sharex=True, squeeze=False, layout="constrained",
                           height_ratios=(1, 1.6)[-len(groups):])
    j = 0
    for row, grp in zip(ax, groups):
        for k, name in grp.items():
            g = E[(E.group == k) & (E.layer == layer)].groupby("year").err
            v = dict(bias=g.mean(), sd=g.std(), rmse=np.sqrt(g.apply(lambda x: (x ** 2).mean())))
            for a, st in zip(row, stats):
                if st == "bias":
                    a.errorbar(v["bias"].index, v["bias"], 2 * v["sd"] / np.sqrt(g.size()), marker="o", ms=3, capsize=2,
                               color=f"C{j}", label=name)
                else:
                    a.plot(v[st].index, v[st], marker="o", ms=3, color=f"C{j}", label=name)
            j += 1
        for a, st in zip(row, stats):
            a.set(ylabel=lab[st][0])
            if st == "bias":
                a.axhline(0, color="#c0392b", lw=0.8)
            else:
                a.set_ylim(0, None)
    if rem is not None and "bias" in stats:
        r = rem.loc[layer]
        ax[-1, stats.index("bias")].plot(r.index, r, "k--", lw=1, label="quadrature remainder")
    for a, st in zip(ax[0], stats):
        a.set_title(lab[st][1])
    fig.legend(*zip(*[(h, l) for a in ax[:, 0] for h, l in zip(*a.get_legend_handles_labels())]),
               loc="outside upper center", ncol=len(labels) + (rem is not None and "bias" in stats), frameon=False,
               fontsize=9)
    return fig


def year_stats(E, labels, layer):
    """Per year and for the 2015-2025 mean (of each replicate's errors): bias, its Monte Carlo se, SD and RMSE over
    replicates, per group."""
    x = E[(E.layer == layer) & E.group.isin(list(labels))]
    x = pd.concat([x.assign(year=x.year.astype(str)), x.groupby(["group", "rep"], as_index=False).err.mean().assign(year="2015-25")])
    g = x.groupby(["year", "group"]).err
    return pd.DataFrame(dict(bias=g.mean(), mc_se=g.sem(), sd=g.std(), rmse=np.sqrt(g.apply(lambda e: (e ** 2).mean()))))


def year_tables(E, labels, layer, bold=("mean_field", "krige", "dr")):
    """Markdown-ready bias (+/- 2 MC se), SD and RMSE tables (years x groups); per row, the `bold` group with the
    smallest |bias| (or SD, RMSE) is bolded."""
    t = year_stats(E, labels, layer)
    out = {}
    for st in ("bias", "sd", "rmse"):
        v = t[st].unstack("group")[list(labels)]
        f = v.map(lambda a: f"{a:+.3f}" if st == "bias" else f"{a:.3f}")
        if st == "bias":
            f = f + " +/- " + (2 * t.mc_se.unstack("group")[list(labels)]).map("{:.3f}".format)
        best = v[list(bold)].abs().idxmin(axis=1)
        for y, k in best.items():
            f.loc[y, k] = f"**{f.loc[y, k]}**"
        f = f.loc[sorted(f.index, key=lambda y: (len(y) > 4, y))]
        out[st] = f.rename(columns=labels).rename_axis(index="year", columns=None)
    return out


def plot_errors(E, labels, layer, split=(), figsize=(11, 5)):
    """Box plots over jittered replicate errors per year and for the 2015-2025 mean (chain.plot_errors)."""
    from osim import chain as ch
    x = E[(E.layer == layer) & E.group.isin(list(labels))]
    x = pd.concat([x.assign(year=x.year.astype(str)),
                   x.groupby(["group", "rep"], as_index=False).err.mean().assign(year="2015-25")])
    w = x.pivot_table(index=["rep", "year"], columns="group", values="err").mul(1e9).reset_index()
    years = sorted(w.year.unique(), key=lambda y: (len(y) > 4, y))
    return ch.plot_errors(w.assign(theta=0.0, box=layer), labels, layer, split, figsize, years)


def qhat_metrics(kind=""):
    """qhat_diag metrics averaged over the diagnostic years."""
    m = pd.read_csv(os.path.join(ns.OUT, DIAG[kind], "metrics.csv"))
    return m.groupby(["sig_deg", "sig_days", "source"]).mean(numeric_only=True).drop(columns="year").reset_index()


def realized_naive(years):
    """Realized naive bias of the total and its Monte Carlo se over the given years (advected floats)."""
    a = pd.read_csv(os.path.join(ns.OUT, "naive_errors.csv"))
    m = a[(a.layer == "total") & a.year.isin(years)].groupby("rep").err.mean()
    return m.mean(), m.std() / np.sqrt(len(m))


def plot_implied_bias(sig_days=30.0):
    """Implied naive bias int Z q-hat / int q-hat - theta of the total against the spatial kernel sd (daily
    positions, domain-restricted), with the realized bias."""
    m = pd.concat([qhat_metrics(""), qhat_metrics("wide")]).query("source == 'daily' and sig_days == @sig_days")
    years = pd.read_csv(os.path.join(ns.OUT, DIAG[""], "metrics.csv")).year.unique()
    b, se = realized_naive(years)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), layout="constrained")
    ax.axhspan(b - 2 * se, b + 2 * se, color="0.85", label="realized bias +/- 2 MC se")
    ax.plot(m.sig_deg, m.bias_naive, "o-", ms=4, label="implied bias")
    ax.set(xscale="log", xlabel="spatial kernel sd (deg)", ylabel="bias (GJ/m2)", ylim=(0, None))
    ax.set_xticks([1 / 12, 1 / 6, 1 / 3, 1 / 2, 1, 2, 3, 5], ["1/12", "1/6", "1/3", "1/2", "1", "2", "3", "5"])
    ax.minorticks_off()
    ax.legend(fontsize=8)
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
        q = q[np.abs(la[:, None] - y).argmin(1)][:, np.abs(lo[:, None] - x).argmin(1)]
        pc = a.pcolormesh(lo, la, np.where(m & (q > 0), q, np.nan), shading="nearest", norm=LogNorm(0.5, 2),
                          cmap="RdBu_r", rasterized=True, transform=ccrs.PlateCarree())
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


COST_STEPS = {"data_download": "GLORYS download (layers, 1000 m velocity)", "domain": "domain mask",
              "truth": "truth theta_y", "advection": "advection (200 replicates)", "ohc_sampling": "OHC at profiles",
              "uniform_comparison": "stationary and iid designs", "cov_exact": "covariance fits (r0-9)",
              "qhat_diag": "q-hat diagnostics", "qhat_crosshalf": "cross-half Hajek", "q_points": "q-hat at profiles",
              "kriging": "kriging + fold maps (r10-109)", "quadrature": "quadrature remainder",
              "dr_bandwidth": "DR kernel sweep (30 kernels)", "dr_layers": "production DR", "spatial_diag": "spatial diagnostics"}


def cost_table(path=None):
    """Per pipeline step (operational jobs, completed tasks): tasks, median and max wall per task, core-h, CPU-h,
    allocated and peak memory, credits billed on open and open-equivalent credits of sc_default work; plus a total."""
    d = pd.read_csv(path or os.path.join(ns.OUT, "compute_costs.csv"))
    d = d.assign(step=d.stage.map(COST_STEPS), wall_h=d.elapsed_s / 3600,
                 credits_open=np.where(d.account.eq("open"), d.credits, 0.0),
                 credits_sc_equiv=np.where(d.account.eq("open"), 0.0, d.credits))
    agg = dict(tasks=("id", "size"), wall_median_h=("wall_h", "median"), wall_max_h=("wall_h", "max"),
               core_h=("core_h", "sum"), cpu_h=("cpu_h", "sum"), mem_alloc_gb=("mem_gb", "max"),
               peak_rss_gb=("maxrss_gb", "max"), credits_open=("credits_open", "sum"),
               credits_sc_equiv=("credits_sc_equiv", "sum"))
    t = d.groupby("step", sort=False).agg(**agg).reindex([v for v in COST_STEPS.values() if v in set(d.step)])
    t = pd.concat([t, d.assign(step="total").groupby("step").agg(**agg)])
    t["credits_total"] = t.credits_open + t.credits_sc_equiv
    return t
