"""Load real spike data from a public DANDI NWB file (streaming, no full download)."""

import numpy as np


DANDISET_ID = "000003"
BIN_SIZE_S = 0.01   # 10 ms bins — matches simulate_spikes timestep semantics
MAX_DURATION_S = 20.0  # cap stream to keep n_timesteps ~2000


def load_nwb_spikes(dandiset_id: str = DANDISET_ID, n_timesteps: int = 2000) -> tuple:
    """
    Pull the first NWB asset with a units table from *dandiset_id* on DANDI,
    bin spike times into a (n_neurons, n_timesteps) float32 array, and return
    that array alongside a dummy state_labels vector of zeros — making this a
    drop-in replacement for simulate_spikes().

    Parameters
    ----------
    dandiset_id : DANDI dandiset identifier, e.g. "000003"
    n_timesteps : target number of time bins (actual value depends on recording)

    Returns
    -------
    spikes       : ndarray (n_neurons, n_timesteps)
    state_labels : ndarray (n_timesteps,) — all zeros
    """
    from dandi.dandiapi import DandiAPIClient
    import remfile
    import h5py
    import pynwb

    print(f"Connecting to DANDI archive, dandiset {dandiset_id} …")
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id)
        asset = _find_nwb_asset(dandiset)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
        print(f"Streaming: {asset.path} ({asset.size / 1e6:.1f} MB)")

    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, "r")
    io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
    try:
        nwb = io.read()
        spikes, state_labels = _bin_units(nwb, n_timesteps)
    finally:
        io.close()

    return spikes, state_labels


def _find_nwb_asset(dandiset):
    """Return the first .nwb asset that contains a units table."""
    for asset in dandiset.get_assets():
        if asset.path.endswith(".nwb"):
            return asset
    raise RuntimeError(
        f"No .nwb file found in dandiset {dandiset.identifier}. "
        "Try a different --dandiset value."
    )


def _bin_units(nwb, n_timesteps: int):
    if nwb.units is None:
        raise RuntimeError(
            "The selected NWB file has no units table. "
            "Choose a dandiset with sorted spike data."
        )

    units = nwb.units
    n_neurons = len(units)
    if n_neurons == 0:
        raise RuntimeError("Units table is empty.")

    spike_times_list = [np.asarray(units["spike_times"][i]) for i in range(n_neurons)]
    all_times = np.concatenate(spike_times_list)
    t_start = float(all_times.min())

    # Cap recording length so we don't stream forever
    t_end = min(float(all_times.max()), t_start + MAX_DURATION_S)
    duration = t_end - t_start
    actual_n_timesteps = min(n_timesteps, max(1, int(duration / BIN_SIZE_S)))

    spikes = np.zeros((n_neurons, actual_n_timesteps), dtype=np.float32)
    for i, times in enumerate(spike_times_list):
        times = times[(times >= t_start) & (times < t_start + actual_n_timesteps * BIN_SIZE_S)]
        bins = ((times - t_start) / BIN_SIZE_S).astype(int)
        bins = np.clip(bins, 0, actual_n_timesteps - 1)
        np.add.at(spikes[i], bins, 1)

    print(f"Loaded {n_neurons} neurons × {actual_n_timesteps} timesteps "
          f"({actual_n_timesteps * BIN_SIZE_S:.1f} s at {BIN_SIZE_S * 1000:.0f} ms bins)")

    state_labels = np.zeros(actual_n_timesteps, dtype=int)
    return spikes, state_labels
