import numpy as np


def simulate_spikes(n_neurons=60, n_timesteps=2000, n_states=4, seed=42):
    """
    Simulate structured neural population activity with distinct behavioral states.

    Neurons form overlapping ensembles tuned to specific behavioral states,
    creating genuine low-dimensional manifold structure in spike space.

    Returns
    -------
    spikes       : ndarray (n_neurons, n_timesteps)  — spike counts
    state_labels : ndarray (n_timesteps,)            — ground-truth state per timestep
    """
    rng = np.random.default_rng(seed)

    # Baseline firing rates (spikes / timestep)
    base_rates = rng.uniform(0.02, 0.06, size=n_neurons)

    # Each state strongly activates a dedicated core ensemble
    ensemble_size = n_neurons // n_states
    state_rates = np.tile(base_rates, (n_states, 1))  # (n_states, n_neurons)
    for s in range(n_states):
        start = s * ensemble_size
        end = start + ensemble_size
        state_rates[s, start:end] *= rng.uniform(5.0, 9.0, size=ensemble_size)

    # Markov state sequence — uniform off-diagonal transitions, realistic dwell times
    transition = np.full((n_states, n_states), 1.0 / (n_states - 1))
    np.fill_diagonal(transition, 0.0)

    state_labels = np.zeros(n_timesteps, dtype=int)
    t, current = 0, 0
    while t < n_timesteps:
        dwell = int(np.clip(rng.exponential(150), 50, 400))
        end = min(t + dwell, n_timesteps)
        state_labels[t:end] = current
        current = rng.choice(n_states, p=transition[current])
        t = end

    # Poisson spikes driven by the current state's rate template
    rates = state_rates[state_labels].T  # (n_neurons, n_timesteps)
    spikes = rng.poisson(rates).astype(np.float32)

    return spikes, state_labels
