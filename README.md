# Neural Representation Explorer

This tool takes simulated brain activity, compresses it down to two dimensions, and reveals that neurons naturally organize into distinct behavioral states — rest, exploration, active movement, and grooming. It shows that complex, high-dimensional neural data has hidden low-dimensional structure that's recoverable with standard machine learning techniques.

> [!NOTE]
> The pipeline auto-runs on every push via **GitHub Actions** and commits the refreshed figures back to this repo — so the images below always reflect the latest code.

---

## How It Works

The pipeline simulates 60 neurons firing over 2000 timesteps, where each neuron's activity is shaped by whichever of four behavioral states (Rest, Explore, Active, Groom) is currently active. A Gaussian kernel (σ = 10) smooths the raw spikes into continuous rate signals. Two dimensionality reduction methods — PCA (linear) and UMAP (neighborhood-preserving) — then compress the 60-dimensional activity into 2-D, where K-Means clustering (k = 4) labels each moment in time with its most likely state.

---

## Visualizations

| Figure | What it shows |
|--------|---------------|
| `manifolds.png` | PCA & UMAP scatter, colored by cluster |
| `trajectory.png` | Population path through state space over time |
| `spike_raster.png` | Raw spikes annotated with behavioral state |
| `firing_rates.png` | Smoothed neural activity heatmap by state |
| `transitions.png` | Empirical Markov transition probability matrix |
| `pca_variance.png` | How much variance each principal component explains |

---

## Results

> Full metrics → [`results/RESULTS.md`](results/RESULTS.md)

### Neural Manifolds — PCA & UMAP

Each behavioral state forms a clean, separated cloud in 2-D.

![Manifolds](results/manifolds.png)

### Neural Trajectory Through State Space

Left panel shows the population path colored by time; right panel colors each point by its ground-truth behavioral state.

![Trajectory](results/trajectory.png)

### Spike Raster

Raw spikes for 40 neurons over 600 timesteps, with the active state marked at the top.

![Spike Raster](results/spike_raster.png)

### Population Firing Rates

Gaussian-smoothed activity across all 60 neurons, where bright bands mark ensemble activation per state.

![Firing Rates](results/firing_rates.png)

### State Transition Matrix

How often each state follows each other state, estimated directly from the simulated sequence.

![Transitions](results/transitions.png)

### PCA Explained Variance

The first few components capture most of the variance, confirming the data lives on a genuinely low-dimensional manifold.

![PCA Variance](results/pca_variance.png)

---

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py
# → results/ contains all figures and summary.json
```

## Project Structure

```
neural_representation_explorer/
├── run_pipeline.py              # full pipeline
├── requirements.txt
├── src/
│   ├── simulate_spikes.py
│   ├── compute_features.py
│   ├── dimensionality.py
│   └── clustering.py
├── notebooks/
│   └── explore_representations.ipynb
└── results/                     # auto-generated
    ├── RESULTS.md
    ├── manifolds.png  trajectory.png  spike_raster.png
    ├── firing_rates.png  transitions.png  pca_variance.png
    └── summary.json
```

To run against a real public Neuropixels recording instead of simulated data, use `python run_pipeline.py --mode real`.
