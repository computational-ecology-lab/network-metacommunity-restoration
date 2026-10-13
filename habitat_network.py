"""Step 1: habitat network for the Snowdonia example (Figure 3F, Figure 5A).

occurrences + environmental rasters
  -> MaxEnt suitability per species
  -> habitat patches per species, merged into shared habitats
  -> carrying capacity K per habitat and species (summed suitability)
  -> least-cost distances between habitats
  -> per-species dispersal matrices  P = exp(-distance / alpha)

Outputs (in outputs/network/):
  habitats.csv, distance_matrix_km.csv, dispersal_<species>.csv,
  fig3F_habitat_network.png, fig5A_H2_H9_corridor.png

Run:  python habitat_network.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, to_hex
from rasterio.warp import Resampling, reproject
from scipy.ndimage import gaussian_filter, label
from skimage.graph import MCP_Geometric
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.abspath(__file__))
OCC_DIR = os.path.join(ROOT, "data", "occurrences")
ENV_DIR = os.path.join(ROOT, "data", "env")
OUT_DIR = os.path.join(ROOT, "outputs", "network")

# species: (occurrence file, common name, label on figures, colour, mean dispersal distance alpha in km)
SPECIES = {
    "Vaccinium myrtillus": ("Vaccinium_myrtillus.csv", "Bilberry", "Species A", "#1f77b4", 2.5),
    "Calluna vulgaris": ("Calluna_vulgaris.csv", "Heather", "Species B", "#9467bd", 1.0),
    "Ulex gallii": ("Ulex_gallii.csv", "Western gorse", "Species C", "#2ca02c", 0.3),
}
BBOX = (-4.2002, 52.8998, -3.6997, 53.2005)  # lon_min, lat_min, lon_max, lat_max
FOCUS = (2, 9)                                # corridor shown in Figure 5A

plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 22,
                     "axes.labelsize": 26, "xtick.labelsize": 20, "ytick.labelsize": 20})


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_rasters():
    """Stack all GeoTIFFs in data/env on the grid of the first one (bilinear)."""
    paths = sorted(os.path.join(ENV_DIR, f) for f in os.listdir(ENV_DIR) if f.endswith(".tif"))
    with rasterio.open(paths[0]) as ref:
        tr, crs, h, w = ref.transform, ref.crs, ref.height, ref.width
    bands = []
    for p in paths:
        with rasterio.open(p) as src:
            if (src.height, src.width) == (h, w) and src.transform == tr:
                a = src.read(1).astype("float32")
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
            else:
                a = np.full((h, w), np.nan, dtype="float32")
                reproject(rasterio.band(src, 1), a, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tr, dst_crs=crs, src_nodata=src.nodata,
                          dst_nodata=np.nan, resampling=Resampling.bilinear)
        bands.append(a)
    data = np.stack(bands)
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return dict(data=data, names=names, nodata=~np.isfinite(data).all(axis=0),
                transform=(tr.c, tr.a, tr.f, -tr.e))   # (x0, px_w, y0, px_h)


def load_occurrences(fname):
    """GBIF export (tab-separated): present records, coordinate uncertainty <= 1 km,
    inside the study box, duplicates removed."""
    df = pd.read_csv(os.path.join(OCC_DIR, fname), sep="\t", on_bad_lines="skip", low_memory=False)
    df = df.rename(columns={"decimalLongitude": "lon", "decimalLatitude": "lat"})
    df = df.dropna(subset=["lon", "lat"])
    df = df[~((df.lon == 0) & (df.lat == 0))]
    df = df[df["occurrenceStatus"].str.upper() == "PRESENT"]
    unc = df["coordinateUncertaintyInMeters"]
    df = df[unc.isna() | (unc <= 1000)]
    df = df[df.lon.between(BBOX[0], BBOX[2]) & df.lat.between(BBOX[1], BBOX[3])]
    df = df.loc[~df[["lon", "lat"]].round(5).duplicated()]
    return df["lon"].to_numpy(), df["lat"].to_numpy()


def to_rowcol(transform, lons, lats):
    x0, pw, y0, ph = transform
    return np.floor((y0 - lats) / ph).astype(int), np.floor((lons - x0) / pw).astype(int)


def to_lonlat(transform, rows, cols):
    x0, pw, y0, ph = transform
    return x0 + (np.asarray(cols) + 0.5) * pw, y0 - (np.asarray(rows) + 0.5) * ph


def pixel_size_m(transform, lat):
    return abs(transform[1]) * 111_320.0 * np.cos(np.radians(lat)), abs(transform[3]) * 111_132.0


# --------------------------------------------------------------------------- #
# MaxEnt (penalised presence-vs-background logistic regression, cloglog output)
# --------------------------------------------------------------------------- #
class Maxent:
    n_knots = 10

    def _expand(self, X):
        hinges = []
        for j, k in enumerate(self.knots):
            hinges += [np.maximum(0, X[:, [j]] - k), np.maximum(0, k - X[:, [j]])]
        return np.hstack([X, X ** 2] + hinges)        # linear, quadratic, hinge features

    def _features(self, X):
        return (self._expand(X) - self.mu) / self.sd

    def fit(self, X_pres, X_bg):
        X_pres = X_pres[np.isfinite(X_pres).all(axis=1)]
        X_bg = X_bg[np.isfinite(X_bg).all(axis=1)]
        X = np.vstack([X_pres, X_bg])
        y = np.r_[np.ones(len(X_pres)), np.zeros(len(X_bg))]
        q = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self.knots = [np.quantile(X[:, j], q) for j in range(X.shape[1])]
        F = self._expand(X)
        self.mu, self.sd = F.mean(axis=0), F.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        self.clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
        self.clf.fit(self._features(X), y, sample_weight=np.ones(len(y)))
        lin = self._linpred(X_bg)
        self.log_mean_bg = lin.max() + np.log(np.mean(np.exp(lin - lin.max())))
        return self

    def _linpred(self, X):
        return self._features(X) @ self.clf.coef_.ravel() + self.clf.intercept_[0]

    def predict(self, X, chunk=200_000):
        out = np.full(len(X), np.nan)
        for i in range(0, len(X), chunk):
            Xc = X[i:i + chunk]
            ok = np.isfinite(Xc).all(axis=1)
            eta = self._linpred(Xc[ok]) - self.log_mean_bg
            out[i:i + chunk][ok] = 1.0 - np.exp(-np.exp(eta))
        return out


def suitability_map(env, lons, lats, n_background=10_000, seed=0):
    d, h, w = env["data"].shape
    flat = env["data"].reshape(d, -1).T
    valid = np.where(~env["nodata"].reshape(-1))[0]
    X_all = flat[valid]

    rows, cols = to_rowcol(env["transform"], lons, lats)
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    X_pres = np.full((len(lons), d), np.nan, dtype="float32")
    X_pres[inside] = env["data"][:, rows[inside], cols[inside]].T
    X_bg = X_all[np.random.default_rng(seed).choice(len(X_all), n_background, replace=False)]

    surface = np.full(h * w, np.nan)
    surface[valid] = Maxent().fit(X_pres, X_bg).predict(X_all)
    return surface.reshape(h, w)


# --------------------------------------------------------------------------- #
# Habitats, resistance and least-cost distances
# --------------------------------------------------------------------------- #
def habitat_patches(suit, density, suit_q=0.85, dens_q=0.5, min_px=50, max_patches=8):
    """Patches = high suitability AND high occurrence density; keep the largest."""
    valid = np.isfinite(suit)
    pos = density[valid & (density > 0)]
    core = valid & (suit >= np.nanquantile(suit, suit_q)) & (density >= np.quantile(pos, dens_q))
    lab, n = label(core)
    sizes = sorted([(i, int((lab == i).sum())) for i in range(1, n + 1)], key=lambda t: t[1], reverse=True)
    sizes = [s for s in sizes if s[1] >= min_px][:max_patches]
    out = np.zeros_like(lab)
    for new, (old, _) in enumerate(sizes, 1):
        out[lab == old] = new
    return out


def merge_habitats(patches, min_px=50):
    """Union of all species' patches; connected regions become shared habitats."""
    lab, n = label(np.any([p > 0 for p in patches.values()], axis=0))
    sizes = sorted([(i, int((lab == i).sum())) for i in range(1, n + 1)], key=lambda t: t[1], reverse=True)
    merged = np.zeros_like(lab)
    contained, n_px = {}, {}
    for new, (old, sz) in enumerate([s for s in sizes if s[1] >= min_px], 1):
        region = lab == old
        merged[region] = new
        contained[new] = sorted(sp for sp, p in patches.items() if np.any((p > 0) & region))
        n_px[new] = sz
    return merged, contained, n_px


def resistance_surface(suit, slope, nodata, rmin=1.0, rmax=100.0):
    def norm(a):
        lo, hi = np.nanpercentile(a[np.isfinite(a)], [2, 98])
        return np.clip((a - lo) / ((hi - lo) if hi > lo else 1.0), 0, 1)
    R = rmin + (rmax - rmin) * ((1 - norm(suit.astype("float64"))) + norm(slope.astype("float64"))) / 2
    R[~np.isfinite(R) | nodata] = rmax
    return R


def least_cost_distances(R, merged, suit, transform, lat):
    """Seed = most suitable cell of each habitat; distance = length of least-cost path."""
    dx, dy = pixel_size_m(transform, lat)
    hids = list(range(1, int(merged.max()) + 1))
    seeds = {h: np.unravel_index(np.argmax(np.where(merged == h, suit, -np.inf)), suit.shape) for h in hids}
    seeds = {h: (int(r), int(c)) for h, (r, c) in seeds.items()}
    dist, paths = np.zeros((len(hids), len(hids))), {}
    for a, ha in enumerate(hids):
        mcp = MCP_Geometric(R, sampling=(dy, dx))
        mcp.find_costs([seeds[ha]])
        for b, hb in enumerate(hids):
            if a != b:
                path = np.array(mcp.traceback(seeds[hb]))
                step = np.diff(path, axis=0).astype("float64")
                dist[a, b] = np.sqrt((step[:, 0] * dy) ** 2 + (step[:, 1] * dx) ** 2).sum()
                paths[(ha, hb)] = path
    return hids, seeds, np.minimum(dist, dist.T), paths


# --------------------------------------------------------------------------- #
# Map (Figure 3F: whole area; Figure 5A: zoom on the H2-H9 corridor)
# --------------------------------------------------------------------------- #
def draw_map(net, fname, window=None, number_size=22, dot_scale=1.0):
    tr, merged = net["transform"], net["merged"]
    h, w = merged.shape
    extent = (tr[0], tr[0] + w * tr[1], tr[2] - h * tr[3], tr[2])
    xmin, xmax, ymin, ymax = window or extent
    n_hab = int(merged.max())
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(net["combined_suit"], extent=extent, cmap="Greys", vmin=0, vmax=1, alpha=0.35, aspect="auto")
    cmap = plt.get_cmap("tab20", n_hab)
    for hid in range(1, n_hab + 1):
        ax.imshow(np.ma.masked_where(merged != hid, np.ones(merged.shape)), extent=extent,
                  cmap=ListedColormap([to_hex(cmap(hid - 1))]), alpha=0.8, aspect="auto")
    ax.contour((merged > 0).astype(float), levels=[0.5], extent=extent, origin="upper",
               colors="black", linewidths=0.6)
    for (a, b), path in net["paths"].items():
        if a < b:
            ax.plot(*to_lonlat(tr, path[:, 0], path[:, 1]), color="#d62728", lw=2.6, zorder=5)
    ddx, ddy = (xmax - xmin) * 0.016, (ymax - ymin) * 0.024
    for hid in range(1, n_hab + 1):
        lon, lat = (float(v) for v in to_lonlat(tr, *net["seeds"][hid]))
        ax.scatter(lon, lat, s=700 * dot_scale, color="white", edgecolors="black", linewidths=1.8, zorder=6)
        ax.text(lon, lat, str(hid), fontsize=number_size, fontweight="bold", ha="center", va="center",
                zorder=7, clip_on=True)
        present = net["contained"][hid]
        for i, sp in enumerate(present):
            ax.scatter(lon + ddx * (i - (len(present) - 1) / 2), lat + ddy, s=260,
                       color=SPECIES[sp][3], edgecolors="black", linewidths=1.2, zorder=8)
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xlabel="Longitude", ylabel="Latitude")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc=v[3], mec="black", ms=16, label=v[2])
                       for v in SPECIES.values()], loc="upper left", fontsize=20, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def zoom_window(net, pad=0.45):
    a, b = FOCUS
    path = net["paths"][(a, b)]
    xs, ys = (list(v) for v in to_lonlat(net["transform"], path[:, 0], path[:, 1]))
    for hid in FOCUS:
        lo, la = to_lonlat(net["transform"], *net["seeds"][hid])
        xs.append(float(lo)); ys.append(float(la))
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    cx, cy, half = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, span * (0.5 + pad)
    return cx - half, cx + half, cy - half, cy + half


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = load_rasters()
    tr = env["transform"]
    h, w = env["nodata"].shape
    lat_c = tr[2] - h * tr[3] / 2

    suit, patches = {}, {}
    for sp, (fname, common, *_rest) in SPECIES.items():
        lons, lats = load_occurrences(fname)
        suit[sp] = suitability_map(env, lons, lats)
        counts = np.zeros((h, w))
        rows, cols = to_rowcol(tr, lons, lats)
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        np.add.at(counts, (rows[ok], cols[ok]), 1.0)
        patches[sp] = habitat_patches(suit[sp], gaussian_filter(counts, sigma=8.0))
        print(f"{common}: {len(lons)} occurrences, {patches[sp].max()} patches")

    merged, contained, n_px = merge_habitats(patches)
    stack = np.stack(list(suit.values()))
    combined = np.full((h, w), np.nan)
    any_val = ~np.all(np.isnan(stack), axis=0)
    combined[any_val] = np.nanmax(stack[:, any_val], axis=0)
    R = resistance_surface(combined, env["data"][env["names"].index("slope")], env["nodata"])
    hids, seeds, dist_m, paths = least_cost_distances(R, merged, combined, tr, lat_c)

    # habitats.csv: species present, area and carrying capacity K (= summed suitability)
    px_m2 = np.prod(pixel_size_m(tr, lat_c))
    rows = []
    for hid in hids:
        region = merged == hid
        row = {"habitat_id": hid, "species_present": "; ".join(SPECIES[s][1] for s in contained[hid]),
               "n_species": len(contained[hid]), "area_ha": round(n_px[hid] * px_m2 / 1e4, 3),
               "pixels": n_px[hid]}
        K = {sp: float(np.nansum(suit[sp][region])) if sp in contained[hid] else 0.0 for sp in SPECIES}
        for sp, v in SPECIES.items():
            row[f"K_{v[1].replace(' ', '_')}"] = round(K[sp], 4)
        row["K_total"] = round(sum(K.values()), 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "habitats.csv"), index=False)

    names = [f"H{i}" for i in hids]
    dist_km = pd.DataFrame(np.round(dist_m / 1000, 4), index=names, columns=names)
    dist_km.to_csv(os.path.join(OUT_DIR, "distance_matrix_km.csv"))
    for v in SPECIES.values():   # negative-exponential dispersal kernel, alpha = mean dispersal distance
        np.exp(-dist_km / v[4]).round(6).to_csv(
            os.path.join(OUT_DIR, f"dispersal_{v[1].replace(' ', '_')}.csv"))

    net = dict(transform=tr, merged=merged, contained=contained, seeds=seeds,
               paths=paths, combined_suit=combined)
    draw_map(net, "fig3F_habitat_network.png")
    plt.rcParams.update({"axes.labelsize": 36, "xtick.labelsize": 28, "ytick.labelsize": 28})
    draw_map(net, "fig5A_H2_H9_corridor.png", window=zoom_window(net), number_size=30, dot_scale=1.43)
    a, b = hids.index(FOCUS[0]), hids.index(FOCUS[1])
    print(f"{len(hids)} habitats; H2-H9 corridor {dist_m[a, b] / 1000:.2f} km; outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
