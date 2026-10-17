"""Démo : débruitage, segmentation à 3 classes et évaluation sur des images
radioscopiques SYNTHÉTIQUES (eau / corium / vapeur).

Les images sont simulées. Aucune donnée KROTOS n'est utilisée.
Usage : python demo.py [--n 20] [--seed 0]
"""
import argparse, csv, os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

WATER, CORIUM, VAPOR = 0, 1, 2
LEVEL = {WATER: 0.55, CORIUM: 0.15, VAPOR: 0.66}   # transmission relative (vapeur peu contrastée)
CMAP = ListedColormap(["#9ecae1", "#8b1a1a", "#fff3b0"])


def make_image(rng, noise, size=256):
    """Renvoie (image bruitée, image propre, vérité terrain à 3 classes)."""
    yy, xx = np.mgrid[:size, :size].astype(float)
    labels = np.full((size, size), WATER, np.uint8)
    # bulles de vapeur : ellipses tournées
    for _ in range(rng.integers(4, 8)):
        cx, cy = rng.uniform(0.15, 0.85) * size, rng.uniform(0.15, 0.9) * size
        rx, ry, a = rng.uniform(10, 26), rng.uniform(14, 34), rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
        v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
        labels[(u / rx) ** 2 + (v / ry) ** 2 <= 1] = VAPOR
    # jet de corium sinueux, puis gouttes
    phase, jet_len = rng.uniform(0, 6.28), rng.uniform(0.55, 0.8) * size
    centre = size * 0.5 + 12 * np.sin(yy / 25 + phase)
    half = 8 + 3 * np.sin(yy / 17 + phase)
    labels[(np.abs(xx - centre) <= half) & (yy <= jet_len)] = CORIUM
    for _ in range(rng.integers(6, 14)):
        cy = rng.uniform(0.3, 0.95) * size
        cx = size * 0.5 + rng.normal(0, 38)
        r = rng.uniform(3, 8)
        labels[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = CORIUM
    # transmission : bulles à bords flous (volume partiel), corium plus net,
    # fond d'eau non uniforme (champ lisse aléatoire) pour mettre à l'épreuve les seuils globaux
    soft_v = gaussian_filter((labels == VAPOR).astype(float), 2.5)
    soft_c = gaussian_filter((labels == CORIUM).astype(float), 1.2)
    drift = gaussian_filter(rng.normal(size=(size, size)), 25)
    drift = 0.035 * drift / drift.std()
    clean = LEVEL[WATER] + drift + (LEVEL[VAPOR] - LEVEL[WATER]) * soft_v
    clean = clean * (1 - soft_c) + LEVEL[CORIUM] * soft_c
    clean += 0.03 * (xx / size - 0.5)                 # éclairage non uniforme
    clean = gaussian_filter(clean, 1.0)               # flou de l'optique
    noisy = np.clip(clean + rng.normal(0, noise, clean.shape), 0, 1)
    return noisy, clean, labels


def denoise(img, method, noise):
    u8 = (img * 255).astype(np.uint8)
    if method == "aucun":
        return img
    if method == "gaussien":
        return cv2.GaussianBlur(img, (0, 0), 1.5)
    if method == "médian":
        return cv2.medianBlur(u8, 5).astype(float) / 255
    if method == "NLM":
        h = max(4.0, 255 * noise * 0.9)
        return cv2.fastNlMeansDenoising(u8, None, h, 7, 21).astype(float) / 255
    raise ValueError(method)


def _ranked(labels3, centres):
    """Classe 0 = niveau le plus bas (corium), 1 = milieu (eau), 2 = haut (vapeur)."""
    order = np.argsort(centres)
    rank = np.empty(3, int)
    rank[order] = np.arange(3)
    return np.array([CORIUM, WATER, VAPOR])[rank[labels3]]


def seg_otsu(img):
    thr = threshold_multiotsu(img, classes=3)
    cls = np.digitize(img, thr)
    return np.array([CORIUM, WATER, VAPOR])[cls]


def seg_kmeans(img):
    x = img.reshape(-1, 1)
    sub = x[np.random.default_rng(0).choice(len(x), 20000, replace=False)]
    km = KMeans(3, n_init=3, random_state=0).fit(sub)
    return _ranked(km.predict(x), km.cluster_centers_.ravel()).reshape(img.shape)


def seg_kmeans_ctx(img):
    """k-means sur [intensité, moyenne locale à 2 échelles] : prend en compte le contexte spatial."""
    f = np.stack([img, gaussian_filter(img, 3), gaussian_filter(img, 8)], -1).reshape(-1, 3)
    sub = f[np.random.default_rng(0).choice(len(f), 20000, replace=False)]
    km = KMeans(3, n_init=3, random_state=0).fit(sub)
    return _ranked(km.predict(f), km.cluster_centers_[:, 0]).reshape(img.shape)


SEGMENTERS = {"Otsu (3 classes)": seg_otsu, "k-means (k=3)": seg_kmeans,
              "k-means + contexte": seg_kmeans_ctx}
RF_NAME = "Forêt aléatoire (supervisé)"
DENOISERS = ["aucun", "gaussien", "médian", "NLM"]


def feats(img):
    """Caractéristiques multi-échelles par pixel (dont fond corrigé et gradient)."""
    g = {s: gaussian_filter(img, s) for s in (1, 2, 4, 8, 16)}
    gy, gx = np.gradient(g[2])
    f = [img, g[1], g[2], g[4], g[8], img - g[16], g[2] - g[8], np.hypot(gx, gy)]
    return np.stack(f, -1).reshape(-1, len(f))


def train_rf(noise, dn, n_train, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_train):
        noisy, _, gt = make_image(rng, noise)
        f = feats(denoise(noisy, dn, noise))
        idx = rng.choice(len(f), 20000, replace=False)
        X.append(f[idx]); y.append(gt.ravel()[idx])
    return RandomForestClassifier(30, max_depth=12, n_jobs=-1, random_state=0).fit(
        np.concatenate(X), np.concatenate(y))


def make_rf_seg(model):
    return lambda img: model.predict(feats(img)).reshape(img.shape)


def dice(pred, gt, k):
    a, b = pred == k, gt == k
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    noises = [0.02, 0.04, 0.06]
    rows = []
    for noise in noises:
        rng = np.random.default_rng(args.seed)
        stats = {}
        # entraînement sur des images distinctes (graine différente) de celles du test
        rfs = {dn: make_rf_seg(train_rf(noise, dn, 6, args.seed + 1000)) for dn in DENOISERS}
        for _ in range(args.n):
            noisy, _, gt = make_image(rng, noise)
            for dn in DENOISERS:
                den = denoise(noisy, dn, noise)
                for sn, sf in {**SEGMENTERS, RF_NAME: rfs[dn]}.items():
                    pred = sf(den)
                    stats.setdefault((dn, sn), []).append(
                        [dice(pred, gt, WATER), dice(pred, gt, CORIUM), dice(pred, gt, VAPOR)])
        for (dn, sn), v in stats.items():
            v = np.array(v)
            rows.append([noise, dn, sn, *v.mean(0).round(3), round(v[:, 2].std(), 3)])
    with open("results/resultats.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["bruit_sigma", "debruitage", "segmentation", "dice_eau",
                    "dice_corium", "dice_vapeur", "ecart_type_vapeur"])
        w.writerows(rows)

    # Figure 1 : exemple du pipeline
    rng = np.random.default_rng(args.seed + 1)
    noisy, clean, gt = make_image(rng, 0.04)
    den = denoise(noisy, "NLM", 0.04)
    panels = [("Vérité terrain", gt, CMAP), ("Image bruitée (σ=0,04)", noisy, "gray"),
              ("Débruitée (NLM)", den, "gray"),
              ("Otsu 3 classes", seg_otsu(den), CMAP), ("k-means k=3", seg_kmeans(den), CMAP),
              ("k-means + contexte", seg_kmeans_ctx(den), CMAP),
              ("Forêt aléatoire", make_rf_seg(train_rf(0.04, "NLM", 6, args.seed + 1000))(den), CMAP)]
    fig, ax = plt.subplots(1, 7, figsize=(22, 3.4))
    for a, (t, im, cm) in zip(ax, panels):
        a.imshow(im, cmap=cm, vmin=0, vmax=2 if cm is CMAP else 1)
        a.set_title(t, fontsize=10); a.axis("off")
    fig.suptitle("Images synthétiques : bleu = eau, rouge = corium, jaune = vapeur", fontsize=11)
    fig.tight_layout(); fig.savefig("results/figure1_pipeline.png", dpi=130); plt.close(fig)

    # Figure 2 : Dice de la vapeur selon le bruit
    fig, ax = plt.subplots(1, 4, figsize=(19, 4), sharey=True)
    for a, sn in zip(ax, [*SEGMENTERS, RF_NAME]):
        for dn in DENOISERS:
            ys = [r[5] for r in rows if r[1] == dn and r[2] == sn]
            a.plot(noises, ys, marker="o", label=dn)
        a.set_title(sn); a.set_xlabel("écart-type du bruit"); a.grid(alpha=.3)
    ax[0].set_ylabel("Dice moyen, classe vapeur"); ax[0].legend(title="débruitage")
    fig.tight_layout(); fig.savefig("results/figure2_dice_vapeur.png", dpi=130); plt.close(fig)

    print("noise  debruitage  segmentation        Dice(eau, corium, vapeur)")
    for r in rows:
        print(f"{r[0]:.2f}  {r[1]:9s}  {r[2]:17s}  {r[3]:.3f} {r[4]:.3f} {r[5]:.3f}")


if __name__ == "__main__":
    main()
