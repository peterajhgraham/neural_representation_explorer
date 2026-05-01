#!/usr/bin/env python3
"""Neural Representation Explorer — full analysis pipeline."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.simulate_spikes import simulate_spikes
from src.compute_features import compute_firing_rates
from src.dimensionality import compute_pca, compute_umap
from src.clustering import cluster_states


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Color palette ──────────────────────────────────────────────────────────────
BG             = "#0d1117"
SURFACE        = "#161b22"
BORDER         = "#30363d"
TEXT           = "#e6edf3"
MUTED          = "#8b949e"
CLUSTER_COLORS = ["#ff6b6b", "#4ecdc4", "#ffd93d", "#a29bfe"]
STATE_NAMES    = ["Rest", "Explore", "Active", "Groom"]


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    SURFACE,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "axes.titlepad":     10,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linewidth":    0.6,
        "legend.facecolor":  SURFACE,
        "legend.edgecolor":  BORDER,
        "font.family":       "monospace",
        "font.size":         10,
    })


def run(mode: str = "sim"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _apply_theme()

    # ── 1. Spike data source ───────────────────────────────────────────────────
    if mode == "real":
        from loaders.nwb_loader import load_nwb_spikes
        spikes, state_labels = load_nwb_spikes()
    else:
        spikes, state_labels = simulate_spikes(n_neurons=60, n_timesteps=2000, n_states=4)
    n_neurons, n_timesteps = spikes.shape

    # ── 2. Gaussian-smoothed firing rates (vectorized, no Python loop) ─────────
    features = compute_firing_rates(spikes, sigma=10)

    # ── 3. Dimensionality reduction ────────────────────────────────────────────
    pca_emb  = compute_pca(features, n_components=2)
    umap_emb = compute_umap(features)

    # ── 4. Cluster neural population states ───────────────────────────────────
    labels = cluster_states(features, k=4)

    # ── 5. Visualize ──────────────────────────────────────────────────────────
    _plot_manifolds(pca_emb, umap_emb, labels)
    _plot_trajectory(umap_emb, state_labels)
    _plot_spike_raster(spikes, state_labels)
    _plot_firing_rate_heatmap(features, state_labels)
    _plot_state_transitions(state_labels, n_states=4)
    _plot_pca_variance(features)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    summary = _build_summary(
        n_neurons=n_neurons, n_timesteps=n_timesteps,
        features=features, labels=labels, state_labels=state_labels,
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_results_markdown(summary)
    print("Pipeline complete. Results in results/")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _state_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=c, label=n) for c, n in zip(CLUSTER_COLORS, STATE_NAMES)]


def _plot_manifolds(pca_emb, umap_emb, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Neural Population Manifolds", color=TEXT, fontsize=13)

    for ax, emb, title, xl, yl in [
        (axes[0], pca_emb,  "PCA",  "PC 1",   "PC 2"),
        (axes[1], umap_emb, "UMAP", "UMAP 1", "UMAP 2"),
    ]:
        for k, (color, name) in enumerate(zip(CLUSTER_COLORS, STATE_NAMES)):
            mask = labels == k
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=color, s=8, alpha=0.75, edgecolors="none", label=name)
        ax.set_title(f"{title}  —  colored by K-Means cluster")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(handles=_state_legend_handles(), markerscale=2,
                  fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "manifolds.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _plot_trajectory(umap_emb, state_labels):
    """UMAP trajectory colored two ways: by time and by ground-truth state."""
    n = len(umap_emb)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Neural Trajectory Through State Space", color=TEXT, fontsize=13)

    # Left: color by time — shows the path the population traces
    sc = axes[0].scatter(
        umap_emb[:, 0], umap_emb[:, 1],
        c=np.arange(n), cmap="plasma", s=6, alpha=0.7, edgecolors="none",
    )
    axes[0].set_title("Colored by time  (plasma = early → late)")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].grid(True, alpha=0.3)
    cb = fig.colorbar(sc, ax=axes[0], label="Timestep")
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=MUTED)

    # Right: color by ground-truth behavioral state
    cmap_states = mcolors.ListedColormap(CLUSTER_COLORS)
    axes[1].scatter(
        umap_emb[:, 0], umap_emb[:, 1],
        c=state_labels, cmap=cmap_states, vmin=0, vmax=3,
        s=6, alpha=0.7, edgecolors="none",
    )
    axes[1].set_title("Colored by ground-truth behavioral state")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(handles=_state_legend_handles(), fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "trajectory.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _plot_spike_raster(spikes, state_labels, max_neurons=40, max_t=600):
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), facecolor=BG,
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0.04},
    )
    fig.suptitle("Spike Raster", color=TEXT, fontsize=13)

    # State timeline strip
    cmap_states = mcolors.ListedColormap(CLUSTER_COLORS)
    axes[0].imshow(
        state_labels[:max_t][None, :], aspect="auto",
        cmap=cmap_states, vmin=0, vmax=3,
        extent=[0, max_t, 0, 1], interpolation="nearest",
    )
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel("State", fontsize=8)
    # mini legend on the state strip
    for i, (c, n) in enumerate(zip(CLUSTER_COLORS, STATE_NAMES)):
        axes[0].text(max_t * (i / len(STATE_NAMES) + 0.02), 0.5, n,
                     color=BG, fontsize=7, va="center", fontweight="bold")

    # Spike raster
    subset = spikes[:max_neurons, :max_t]
    for nidx in range(max_neurons):
        spike_times = np.where(subset[nidx] > 0)[0]
        axes[1].scatter(spike_times, np.full_like(spike_times, nidx),
                        s=1.5, color="#58a6ff", alpha=0.55)
    axes[1].set_xlim(0, max_t)
    axes[1].set_ylim(-1, max_neurons)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Neuron index")
    axes[1].grid(True, alpha=0.2)

    fig.savefig(os.path.join(RESULTS_DIR, "spike_raster.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _plot_firing_rate_heatmap(features, state_labels):
    n_t = features.shape[0]
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), facecolor=BG,
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0.04},
    )
    fig.suptitle("Population Firing Rates", color=TEXT, fontsize=13)

    # State timeline strip
    cmap_states = mcolors.ListedColormap(CLUSTER_COLORS)
    axes[0].imshow(
        state_labels[:n_t][None, :], aspect="auto",
        cmap=cmap_states, vmin=0, vmax=3,
        extent=[0, n_t, 0, 1], interpolation="nearest",
    )
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel("State", fontsize=8)

    # Firing rate heatmap
    im = axes[1].imshow(
        features.T, aspect="auto", cmap="inferno",
        interpolation="nearest", origin="lower",
    )
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Neuron index")
    cb = fig.colorbar(im, ax=axes[1], label="Smoothed firing rate")
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=MUTED)

    fig.savefig(os.path.join(RESULTS_DIR, "firing_rates.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _plot_state_transitions(state_labels, n_states=4):
    """Empirical Markov transition matrix — how often does each state follow each other?"""
    T = np.zeros((n_states, n_states), dtype=int)
    for a, b in zip(state_labels[:-1], state_labels[1:]):
        T[a, b] += 1

    row_sums = T.sum(axis=1, keepdims=True).clip(min=1)
    T_prob = T / row_sums

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    im = ax.imshow(T_prob, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("State Transition Probabilities")
    for i in range(n_states):
        for j in range(n_states):
            ax.text(j, i, f"{T_prob[i, j]:.2f}",
                    ha="center", va="center", color=TEXT, fontsize=11)
    cb = fig.colorbar(im, ax=ax, label="Probability")
    cb.ax.yaxis.label.set_color(TEXT)
    cb.ax.tick_params(colors=MUTED)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "transitions.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _plot_pca_variance(features):
    from sklearn.decomposition import PCA
    pca_full = PCA().fit(features)
    ind_var = pca_full.explained_variance_ratio_ * 100
    cum_var = np.cumsum(ind_var)

    n_show = min(20, len(ind_var))
    x = np.arange(1, n_show + 1)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    ax.bar(x, ind_var[:n_show], color="#4ecdc4", alpha=0.85, label="Individual")

    ax2 = ax.twinx()
    ax2.plot(x, cum_var[:n_show], "o-", color="#ffd93d",
             markersize=5, linewidth=2, label="Cumulative")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.tick_params(colors=MUTED)
    ax2.yaxis.label.set_color(TEXT)
    for spine in ax2.spines.values():
        spine.set_edgecolor(BORDER)

    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Individual variance (%)")
    ax.grid(True, alpha=0.3)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pca_variance.png"),
                dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── Summary ───────────────────────────────────────────────────────────────────

def _build_summary(*, n_neurons, n_timesteps, features, labels, state_labels):
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA

    sil = float(silhouette_score(features, labels))

    pca_full = PCA().fit(features)
    dims_90  = int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.90) + 1)

    k = int(labels.max()) + 1
    cluster_sizes = {int(i): int(np.sum(labels == i)) for i in range(k)}

    # Dwell-time statistics
    dwells, t, cur = [], 0, state_labels[0]
    for i, s in enumerate(state_labels):
        if s != cur:
            dwells.append(i - t)
            t, cur = i, s
    dwells.append(len(state_labels) - t)

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parameters": {
            "n_neurons":        n_neurons,
            "n_timesteps":      n_timesteps,
            "smoothing_sigma":  10,
            "n_clusters":       k,
        },
        "results": {
            "n_timepoints":              int(features.shape[0]),
            "feature_dimensions":        int(features.shape[1]),
            "silhouette_score":          round(sil, 4),
            "pca_dims_for_90pct_variance": dims_90,
            "cluster_sizes":             cluster_sizes,
            "mean_state_dwell_timesteps": round(float(np.mean(dwells)), 1),
        },
    }


def _write_results_markdown(summary):
    p = summary["parameters"]
    r = summary["results"]

    rows = "".join(
        f"| {cid} | {STATE_NAMES[cid]} | {size} |\n"
        for cid, size in r["cluster_sizes"].items()
    )

    md = f"""# Pipeline Results

> Auto-generated on {summary['timestamp'][:10]}

## Parameters

| Parameter | Value |
|-----------|-------|
| Neurons | {p['n_neurons']} |
| Timesteps | {p['n_timesteps']} |
| Smoothing σ | {p['smoothing_sigma']} |
| Clusters (k) | {p['n_clusters']} |

## Metrics

| Metric | Value |
|--------|-------|
| Silhouette score | **{r['silhouette_score']}** |
| PCA dims → 90% variance | {r['pca_dims_for_90pct_variance']} |
| Mean state dwell | {r['mean_state_dwell_timesteps']} timesteps |

## Cluster Distribution

| Cluster | State | Size |
|---------|-------|------|
{rows}
## Neural Manifolds

![Manifolds](manifolds.png)

## Neural Trajectory

![Trajectory](trajectory.png)

## Spike Raster

![Spike Raster](spike_raster.png)

## Population Firing Rates

![Firing Rates](firing_rates.png)

## State Transition Matrix

![Transitions](transitions.png)

## PCA Explained Variance

![PCA Variance](pca_variance.png)
"""
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Neural Representation Explorer")
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="'sim' uses simulated spikes (default); 'real' streams a public NWB file from DANDI",
    )
    args = parser.parse_args()
    run(mode=args.mode)
