import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
Interhemispheric beta-band functional connectivity on the MNE somato MEG dataset.

This script reproduces the logic of an EEG sliding-window FC analysis using:
- 500 ms windows
- 50 ms step size (90% overlap)
- coherence and wPLI
- beta band: 13-30 Hz (1-Hz steps)

For simplicity, this version uses ONE left-hemisphere and ONE right-hemisphere
MEG gradiometer, intended as rough sensor-level proxies for left/right M1.

Later, this can be extended to ROI/cluster averages or source-space M1 signals.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import mne

    from pathlib import Path
    from mne.datasets import somato
    from mne_connectivity import spectral_connectivity_epochs

    return Path, mne, mo, np, plt, somato, spectral_connectivity_epochs


@app.cell
def _(Path, np):
    # =============================================================================
    # Configuration
    # =============================================================================

    # -------------------------
    # Dataset / epoch settings
    # -------------------------
    SUBJECT = "01"
    TASK = "somato"

    CROP_TMIN = 120.0
    CROP_TMAX = 360.0
    RESAMPLE_SFREQ = 200.0

    EVENT_ID = 1
    EPOCH_TMIN = -1.0
    EPOCH_TMAX = 3.0

    # -------------------------
    # Sliding-window settings
    # -------------------------
    WIN_LEN_SEC = 0.5      # 500 ms
    STEP_SEC = 0.05        # 50 ms (90% overlap)

    # -------------------------
    # Frequency settings
    # -------------------------
    FREQS = np.arange(13.0, 31.0, 1.0)   # 13-30 Hz inclusive
    FMIN = 13.0
    FMAX = 30.0

    # -------------------------
    # Connectivity methods
    # -------------------------
    # MNE names:
    #   'coh'  = coherence
    #   'wpli' = weighted phase lag index
    METHODS = ["coh", "wpli"]

    # -------------------------
    # Channel selection
    # -------------------------
    # Option 1: manually set known channel names here (recommended after inspection)
    LEFT_CH_NAME = None
    RIGHT_CH_NAME = None

    # Option 2: if None above, choose automatically using sensor positions
    # as rough proxies for left/right central-motor gradiometers.
    AUTO_SELECT_VERBOSE = True

    # -------------------------
    # Output
    # -------------------------
    OUT_DIR = Path("somato_fc_results")
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    return (
        AUTO_SELECT_VERBOSE,
        EPOCH_TMAX,
        EPOCH_TMIN,
        EVENT_ID,
        FMAX,
        FMIN,
        FREQS,
        LEFT_CH_NAME,
        METHODS,
        OUT_DIR,
        RESAMPLE_SFREQ,
        RIGHT_CH_NAME,
        STEP_SEC,
        SUBJECT,
        TASK,
        WIN_LEN_SEC,
    )


@app.cell
def _(FMAX, FMIN, mne, np, plt, spectral_connectivity_epochs):
    # =============================================================================
    # Utility functions
    # =============================================================================

    def select_left_right_motor_like_grads(info, verbose=True):
        """
        Automatically select one left and one right MEG gradiometer as rough
        proxies for left/right sensorimotor cortex.

        Heuristic:
        - use grad channels only
        - favor channels with:
            * negative x (left) / positive x (right)
            * y close to 0 (central AP position)
            * high z (superior/top of helmet)
            * moderate lateral x (not too medial, not too lateral)

        This is only a pragmatic sensor-level approximation.
        """
        grad_picks = mne.pick_types(info, meg="grad", eeg=False, eog=False, stim=False)
        ch_names = [info["ch_names"][p] for p in grad_picks]

        pos = []
        for p in grad_picks:
            loc = info["chs"][p]["loc"][:3]
            pos.append(loc)
        pos = np.array(pos)

        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        # Target lateral distance (in meters) chosen heuristically
        x_target = 0.04
        z_max = np.nanmax(z)

        # Score function: smaller is better
        # Favor:
        # - central AP (small |y|)
        # - near x_target laterality
        # - superior sensors (z close to z_max)
        def score(mask):
            idx = np.where(mask)[0]
            s = (
                1.0 * np.abs(y[idx]) +
                0.8 * np.abs(np.abs(x[idx]) - x_target) +
                0.6 * (z_max - z[idx])
            )
            best_local = idx[np.argmin(s)]
            return best_local

        # Left: x < 0, Right: x > 0
        left_idx = score(x < 0)
        right_idx = score(x > 0)

        left_ch = ch_names[left_idx]
        right_ch = ch_names[right_idx]

        if verbose:
            print("\n[Auto channel selection]")
            print(f"Selected left  sensor: {left_ch}")
            print(f"Selected right sensor: {right_ch}")

        return left_ch, right_ch

    def get_two_channel_data(epochs, left_ch, right_ch):
        """Return data array of shape (n_epochs, 2, n_times)."""
        ep = epochs.copy().pick([left_ch, right_ch])
        data = ep.get_data()  # (n_epochs, 2, n_times)
        return data

    def make_sliding_windows(tmin, tmax, win_len, step, sfreq, n_times):
        """
        Create overlapping windows.

        Returns
        -------
        windows : list of dict
            Each dict contains:
            - 'start_sec'
            - 'stop_sec'
            - 'center_sec'
            - 'start_samp'
            - 'stop_samp'
        """
        starts_sec = np.arange(tmin, tmax - win_len + 1e-12, step)
        windows = []

        for s in starts_sec:
            e = s + win_len

            # Convert to sample indices relative to epoch start
            start_samp = int(round((s - tmin) * sfreq))
            stop_samp = int(round((e - tmin) * sfreq))

            # Guard against boundary problems
            if start_samp < 0 or stop_samp > n_times:
                continue

            windows.append(
                dict(
                    start_sec=float(s),
                    stop_sec=float(e),
                    center_sec=float(s + win_len / 2.0),
                    start_samp=start_samp,
                    stop_samp=stop_samp,
                )
            )

        return windows

    def compute_connectivity_per_window(data_2ch, sfreq, windows, freqs, methods):
        """
        Compute FC for each window for the single left-right pair.

        Parameters
        ----------
        data_2ch : ndarray, shape (n_epochs, 2, n_times)
        sfreq : float
        windows : list of dicts
        freqs : ndarray
        methods : list of str

        Returns
        -------
        results : dict
            results[method] -> array, shape (n_windows, n_freqs)
        """
        results = {m: [] for m in methods}
        indices = (np.array([0]), np.array([1]))  # only L-R connection

        for wi, w in enumerate(windows):
            seg = data_2ch[:, :, w["start_samp"]:w["stop_samp"]]

            for method in methods:
                con = spectral_connectivity_epochs(
                    data=seg,
                    method=method,
                    mode="fourier",
                    sfreq=sfreq,
                    cwt_freqs=None,
                    fmin=FMIN,
                    fmax=FMAX,
                    faverage=False,
                    indices=indices,
                    verbose=False,
                )

                # Expected data shape for one connection:
                # (1, n_freqs) or possibly squeezed equivalent depending on version
                cdat = np.asarray(con.get_data())
                cdat = np.squeeze(cdat)

                # Ensure final shape is (n_freqs,)
                if cdat.ndim != 1:
                    raise RuntimeError(
                        f"Unexpected connectivity output shape for method={method}: {cdat.shape}"
                    )

                results[method].append(cdat)

            if (wi + 1) % 10 == 0 or (wi + 1) == len(windows):
                print(f"Processed {wi + 1}/{len(windows)} windows")

        results = {m: np.vstack(v) for m, v in results.items()}  # (n_windows, n_freqs)
        return results

    def save_results_npz(out_file, times, freqs, raw_results, meta):
        np.savez(
            out_file,
            times=times,
            freqs=freqs,
            raw_coh=raw_results.get("coh"),
            raw_wpli=raw_results.get("wpli"),
            meta=meta,
        )

    def save_csv_per_method(out_dir, times, freqs, arr, prefix):
        """
        Save one CSV file per method in a simple tabular format:
        first column = time, remaining columns = frequencies.
        """
        header = "time_sec," + ",".join([f"{f:.1f}Hz" for f in freqs])
        csv_arr = np.column_stack([times, arr])
        np.savetxt(out_dir / f"{prefix}.csv", csv_arr, delimiter=",", header=header, comments="")

    def plot_time_frequency(times, freqs, arr, title, out_file):
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(
            arr.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Connectivity")
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        return fig

    def plot_beta_average(times, arr, freqs, title, out_file):
        beta_mean = arr.mean(axis=1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, beta_mean, linewidth=2)
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean beta connectivity")
        ax.set_title(title + f" ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz avg)")
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        return fig

    return (
        compute_connectivity_per_window,
        get_two_channel_data,
        make_sliding_windows,
        plot_beta_average,
        plot_time_frequency,
        save_csv_per_method,
        save_results_npz,
        select_left_right_motor_like_grads,
    )


@app.cell
def _(RESAMPLE_SFREQ, SUBJECT, TASK, mne, somato):
    # =============================================================================
    # Main
    # =============================================================================
    print("Loading somato dataset...")
    data_path = somato.data_path()
    raw_fname = data_path / f"sub-{SUBJECT}" / "meg" / f"sub-{SUBJECT}_task-{TASK}_meg.fif"

    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

    print("Cropping, loading, and resampling raw data...")
    # raw.crop(CROP_TMIN, CROP_TMAX).resample(RESAMPLE_SFREQ)
    raw.resample(RESAMPLE_SFREQ)
    raw
    return (raw,)


@app.cell
def _(EPOCH_TMAX, EPOCH_TMIN, EVENT_ID, mne, raw):
    print("Finding events...")
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)

    picks = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, stim=False)

    print("Constructing epochs...")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=picks,
        baseline=(None, 0),
        preload=True,
        reject=None,
        verbose=False,
    )

    print(f"Number of epochs: {len(epochs)}")
    print(f"Epoch shape: {epochs.get_data().shape}")
    epochs
    return (epochs,)


@app.cell
def _(
    AUTO_SELECT_VERBOSE,
    LEFT_CH_NAME,
    RIGHT_CH_NAME,
    epochs,
    select_left_right_motor_like_grads,
):
    # -------------------------------------------------------------------------
    # Select one left and one right channel
    # -------------------------------------------------------------------------
    left_ch = LEFT_CH_NAME
    right_ch = RIGHT_CH_NAME

    if left_ch is None or right_ch is None:
        left_ch, right_ch = select_left_right_motor_like_grads(
            epochs.info, verbose=AUTO_SELECT_VERBOSE
        )

    print("\nUsing channels:")
    print(f"  Left  hemisphere proxy: {left_ch}")
    print(f"  Right hemisphere proxy: {right_ch}")
    return left_ch, right_ch


@app.cell
def _(
    EPOCH_TMAX,
    EPOCH_TMIN,
    STEP_SEC,
    WIN_LEN_SEC,
    epochs,
    get_two_channel_data,
    left_ch,
    make_sliding_windows,
    np,
    right_ch,
):
    data_2ch = get_two_channel_data(epochs, left_ch, right_ch)
    sfreq = epochs.info["sfreq"]
    n_epochs, n_ch, n_times = data_2ch.shape

    print(f"\nTwo-channel data shape: {data_2ch.shape}")

    # -------------------------------------------------------------------------
    # Create sliding windows
    # -------------------------------------------------------------------------
    windows = make_sliding_windows(
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        win_len=WIN_LEN_SEC,
        step=STEP_SEC,
        sfreq=sfreq,
        n_times=n_times,
    )

    times = np.array([w["center_sec"] for w in windows])

    print(f"\nNumber of sliding windows: {len(windows)}")
    print(f"First window: {windows[0]['start_sec']:.2f} to {windows[0]['stop_sec']:.2f} s")
    print(f"Last  window: {windows[-1]['start_sec']:.2f} to {windows[-1]['stop_sec']:.2f} s")
    return data_2ch, sfreq, times, windows


@app.cell
def _(FREQS, METHODS, compute_connectivity_per_window, data_2ch, sfreq, windows):
    # -------------------------------------------------------------------------
    # Compute connectivity
    # -------------------------------------------------------------------------
    print("\nComputing sliding-window connectivity...")
    raw_results = compute_connectivity_per_window(
        data_2ch=data_2ch,
        sfreq=sfreq,
        windows=windows,
        freqs=FREQS,
        methods=METHODS,
    )
    return (raw_results,)


@app.cell
def _(
    EPOCH_TMAX,
    EPOCH_TMIN,
    FREQS,
    METHODS,
    OUT_DIR,
    STEP_SEC,
    SUBJECT,
    TASK,
    WIN_LEN_SEC,
    left_ch,
    raw_results,
    right_ch,
    save_csv_per_method,
    save_results_npz,
    times,
):
    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    meta = {
        "subject": SUBJECT,
        "task": TASK,
        "left_ch": left_ch,
        "right_ch": right_ch,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "win_len_sec": WIN_LEN_SEC,
        "step_sec": STEP_SEC,
        "freqs_hz": FREQS.tolist(),
    }

    save_results_npz(
        OUT_DIR / "somato_interhemispheric_fc.npz",
        times=times,
        freqs=FREQS,
        raw_results=raw_results,
        meta=meta,
    )

    for method in METHODS:
        save_csv_per_method(OUT_DIR, times, FREQS, raw_results[method], f"raw_{method}")
    return


@app.cell
def _(
    FREQS,
    METHODS,
    OUT_DIR,
    left_ch,
    mo,
    plot_beta_average,
    plot_time_frequency,
    raw_results,
    right_ch,
    times,
):
    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    _figs = []
    for _method in METHODS:
        _figs.append(
            plot_time_frequency(
                times,
                FREQS,
                raw_results[_method],
                title=f"Raw {_method} connectivity ({left_ch} ↔ {right_ch})",
                out_file=OUT_DIR / f"raw_{_method}_time_frequency.png",
            )
        )

        _figs.append(
            plot_beta_average(
                times,
                raw_results[_method],
                FREQS,
                title=f"Raw {_method}",
                out_file=OUT_DIR / f"raw_{_method}_beta_average.png",
            )
        )

    print("\nDone.")
    print(f"Results saved in: {OUT_DIR.resolve()}")
    mo.vstack(_figs)
    return


if __name__ == "__main__":
    app.run()
