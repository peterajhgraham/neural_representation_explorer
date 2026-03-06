#!/usr/bin/env python3
"""Run the full neural representation explorer pipeline and save results."""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.simulate_spikes import simulate_spikes
from src.compute_features import compute_firing_rates
from src.dimensionality import compute_pca, compute_umap
from src.clustering import cluster_states


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(42)

    # --- 1. Simulate spikes ---
    n_neurons, n_timesteps = 50, 1000
    spikes = simulate_spikes(n_neurons=n_neurons, n_timesteps=n_timesteps)

    # --- 2. Compute features ---
    window = 20
    features = compute_firing_rates(spikes, window=window)

    # --- 3. Dimensionality reduction ---
    pca_embedding = compute_pca(features, n_components=2)
    umap_embedding = compute_umap(features)

    # --- 4. Clustering ---
    k = 4
    labels = cluster_states(features, k=k)

    # --- 5. Generate figures ---
    _plot_manifolds(pca_embedding, umap_embedding, labels)
    _plot_spike_raster(spikes)
    _plot_firing_rate_heatmap(features)
    _plot_cluster_distribution(labels, k)
    _plot_pca_variance(features)

    # --- 6. Write summary ---
    summary = _build_summary(
        n_neurons=n_neurons,
        n_timesteps=n_timesteps,
        window=window,
        k=k,
        features=features,
        labels=labels,
        pca_embedding=pca_embedding,
        umap_embedding=umap_embedding,
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _write_results_markdown(summary)

    print("Pipeline complete. Results written to results/")


# ---------- plotting helpers ----------

def _plot_manifolds(pca_emb, umap_emb, labels):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(
        pca_emb[:, 0], pca_emb[:, 1],
        c=labels, cmap="viridis", s=12, alpha=0.8, edgecolors="none",
    )
    axes[0].set_title("PCA — Neural Population States")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    plt.colorbar(sc0, ax=axes[0], label="Cluster")

    sc1 = axes[1].scatter(
        umap_emb[:, 0], umap_emb[:, 1],
        c=labels, cmap="viridis", s=12, alpha=0.8, edgecolors="none",
    )
    axes[1].set_title("UMAP — Neural Population States")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    plt.colorbar(sc1, ax=axes[1], label="Cluster")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "manifolds.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_spike_raster(spikes, max_neurons=30, max_timesteps=300):
    fig, ax = plt.subplots(figsize=(12, 5))
    subset = spikes[:max_neurons, :max_timesteps]
    for neuron_idx in range(subset.shape[0]):
        spike_times = np.where(subset[neuron_idx] > 0)[0]
        ax.scatter(spike_times, np.full_like(spike_times, neuron_idx),
                   s=1, color="black")
    ax.set_title("Spike Raster (first 30 neurons, first 300 time steps)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Neuron")
    ax.set_ylim(-1, max_neurons)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "spike_raster.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_firing_rate_heatmap(features):
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(features.T, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_title("Firing Rate Heatmap (neurons × time windows)")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Neuron")
    plt.colorbar(im, ax=ax, label="Firing rate")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "firing_rates.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_distribution(labels, k):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [np.sum(labels == i) for i in range(k)]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, k))
    ax.bar(range(k), counts, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of time windows")
    ax.set_xticks(range(k))
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cluster_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_variance(features):
    from sklearn.decomposition import PCA
    pca_full = PCA().fit(features)
    explained = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    n_show = min(20, len(explained))
    ax.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show] * 100,
           color="steelblue", alpha=0.8, label="Individual")
    ax.plot(range(1, n_show + 1), explained[:n_show],
            "o-", color="darkorange", markersize=5, label="Cumulative")
    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "pca_variance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- summary helpers ----------

def _build_summary(*, n_neurons, n_timesteps, window, k, features, labels,
                   pca_embedding, umap_embedding):
    from sklearn.metrics import silhouette_score
    sil = float(silhouette_score(features, labels))

    from sklearn.decomposition import PCA
    pca_full = PCA().fit(features)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    dims_90 = int(np.searchsorted(cumvar, 0.90) + 1)

    cluster_sizes = {int(i): int(np.sum(labels == i)) for i in range(k)}

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "parameters": {
            "n_neurons": n_neurons,
            "n_timesteps": n_timesteps,
            "window_size": window,
            "n_clusters": k,
        },
        "results": {
            "n_time_windows": int(features.shape[0]),
            "feature_dimensions": int(features.shape[1]),
            "silhouette_score": round(sil, 4),
            "pca_dims_for_90pct_variance": dims_90,
            "cluster_sizes": cluster_sizes,
        },
    }


def _write_results_markdown(summary):
    p = summary["parameters"]
    r = summary["results"]

    md = f"""# Results

> Auto-generated by `run_pipeline.py` on {summary['timestamp'][:10]}

## Parameters

| Parameter | Value |
|-----------|-------|
| Neurons | {p['n_neurons']} |
| Time steps | {p['n_timesteps']} |
| Window size | {p['window_size']} |
| Clusters (k) | {p['n_clusters']} |

## Key Metrics

| Metric | Value |
|--------|-------|
| Time windows | {r['n_time_windows']} |
| Feature dimensions | {r['feature_dimensions']} |
| Silhouette score | {r['silhouette_score']} |
| PCA dims for 90% variance | {r['pca_dims_for_90pct_variance']} |

## Cluster Distribution

| Cluster | Size |
|---------|------|
""" + "".join(
        f"| {cid} | {size} |\n"
        for cid, size in r["cluster_sizes"].items()
    ) + f"""
## Spike Raster

![Spike Raster](spike_raster.png)

## Firing Rate Heatmap

![Firing Rates](firing_rates.png)

## PCA Explained Variance

![PCA Variance](pca_variance.png)

## Neural Manifolds (PCA & UMAP)

![Manifolds](manifolds.png)

## Cluster Size Distribution

![Cluster Distribution](cluster_distribution.png)
"""
    with open(os.path.join(RESULTS_DIR, "RESULTS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    run()
