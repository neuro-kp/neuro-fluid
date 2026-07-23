import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Interhemispheric beta-band functional connectivity

    A marimo port of `scripts/meg_connectivity_analysis_somato_dataset.py`, run on the
    MNE `somato` MEG dataset. The pipeline is broken into steps so each intermediate
    (events, epochs, selected channels, sliding windows, per-window connectivity) can be
    inspected inline, and the final views are reactive.

    Pipeline:

    - 500 ms sliding windows, 50 ms step (90% overlap)
    - coherence and wPLI
    - beta band restricted to 13–30 Hz
    - one left- and one right-hemisphere gradiometer as rough proxies for left/right M1

    These sensor-level proxies are a stand-in for the ROI/cluster averages that come later
    (issues #10, #11).
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import mne

    from pathlib import Path
    from mne.datasets import somato
    from mne_connectivity import spectral_connectivity_epochs

    return Path, mne, mo, np, pd, plt, somato, spectral_connectivity_epochs


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration
    """)
    return


@app.cell
def _():
    SUBJECT = "01"
    TASK = "somato"

    RESAMPLE_SFREQ = 200.0

    EVENT_ID = 1
    EPOCH_TMIN = -1.0
    EPOCH_TMAX = 3.0

    WIN_LEN_SEC = 0.5
    STEP_SEC = 0.05  # 90% overlap

    # At 200 Hz, a 0.5 s window gives 2 Hz Fourier resolution, so connectivity bins are 2 Hz, not 1.
    FMIN = 13.0
    FMAX = 30.0

    # 'coh' = coherence, 'wpli' = weighted phase lag index
    METHODS = ["coh", "wpli"]
    return (
        EPOCH_TMAX,
        EPOCH_TMIN,
        EVENT_ID,
        FMAX,
        FMIN,
        METHODS,
        RESAMPLE_SFREQ,
        STEP_SEC,
        SUBJECT,
        TASK,
        WIN_LEN_SEC,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Helper functions

    Channel selection, window construction, and the per-window connectivity loop, lifted
    from the original script. `compute_connectivity_per_window` additionally returns the
    actual frequency bins produced by the Fourier transform.
    """)
    return


@app.cell
def _(mne, np, spectral_connectivity_epochs):
    def select_left_right_motor_like_grads(info, verbose=True):
        """Pick one left and one right MEG gradiometer as rough proxies for left/right
        sensorimotor cortex, favouring central-anterior, superior, moderately lateral sensors."""
        grad_picks = mne.pick_types(info, meg="grad", eeg=False, eog=False, stim=False)
        ch_names = [info["ch_names"][p] for p in grad_picks]

        pos = np.array([info["chs"][p]["loc"][:3] for p in grad_picks])
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        x_target = 0.04
        z_max = np.nanmax(z)

        def score(mask):
            idx = np.where(mask)[0]
            s = (
                1.0 * np.abs(y[idx])
                + 0.8 * np.abs(np.abs(x[idx]) - x_target)
                + 0.6 * (z_max - z[idx])
            )
            return idx[np.argmin(s)]

        left_ch = ch_names[score(x < 0)]
        right_ch = ch_names[score(x > 0)]

        if verbose:
            print(f"Selected left  sensor: {left_ch}")
            print(f"Selected right sensor: {right_ch}")

        return left_ch, right_ch

    def get_two_channel_data(epochs, left_ch, right_ch):
        """Return data array of shape (n_epochs, 2, n_times)."""
        return epochs.copy().pick([left_ch, right_ch]).get_data()

    def make_sliding_windows(tmin, tmax, win_len, step, sfreq, n_times):
        """Overlapping windows as a list of dicts with second and sample bounds."""
        starts_sec = np.arange(tmin, tmax - win_len + 1e-12, step)
        windows = []
        for s in starts_sec:
            e = s + win_len
            start_samp = int(round((s - tmin) * sfreq))
            stop_samp = int(round((e - tmin) * sfreq))
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

    def compute_connectivity_per_window(data_2ch, sfreq, windows, fmin, fmax, methods):
        """FC for each window for the single left-right pair.

        Returns (results, freqs) where results[method] has shape (n_windows, n_freqs) and
        freqs are the actual Fourier frequency bins within [fmin, fmax]."""
        results = {m: [] for m in methods}
        indices = (np.array([0]), np.array([1]))  # only L-R connection
        freqs_actual = None

        for w in windows:
            seg = data_2ch[:, :, w["start_samp"] : w["stop_samp"]]
            for method in methods:
                con = spectral_connectivity_epochs(
                    data=seg,
                    method=method,
                    mode="fourier",
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    faverage=False,
                    indices=indices,
                    verbose=False,
                )
                if freqs_actual is None:
                    freqs_actual = np.asarray(con.freqs)
                cdat = np.squeeze(np.asarray(con.get_data()))
                if cdat.ndim != 1:
                    raise RuntimeError(
                        f"Unexpected connectivity output shape for method={method}: {cdat.shape}"
                    )
                results[method].append(cdat)

        results = {m: np.vstack(v) for m, v in results.items()}
        return results, freqs_actual

    return (
        compute_connectivity_per_window,
        get_two_channel_data,
        make_sliding_windows,
        select_left_right_motor_like_grads,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the raw data

    Downloads the `somato` dataset on first run (cached afterwards by MNE), then resamples
    to 200 Hz.
    """)
    return


@app.cell
def _(RESAMPLE_SFREQ, SUBJECT, TASK, mne, somato):
    _data_path = somato.data_path()
    _raw_fname = _data_path / f"sub-{SUBJECT}" / "meg" / f"sub-{SUBJECT}_task-{TASK}_meg.fif"

    raw = mne.io.read_raw_fif(_raw_fname, preload=True, verbose=False)
    raw.resample(RESAMPLE_SFREQ)
    raw
    return (raw,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Events
    """)
    return


@app.cell
def _(mne, pd, raw):
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    events_df = pd.DataFrame(events, columns=["sample", "prev_id", "event_id"])
    events_df
    return (events,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Epochs

    Cut trials from -1 s to +3 s around each cue, baseline-corrected against the pre-cue
    interval, on gradiometer channels only. The shape is (n_epochs, n_channels, n_times).
    """)
    return


@app.cell
def _(EPOCH_TMAX, EPOCH_TMIN, EVENT_ID, events, mne, raw):
    _picks = mne.pick_types(raw.info, meg="grad", eeg=False, eog=False, stim=False)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=_picks,
        baseline=(None, 0),
        preload=True,
        reject=None,
        verbose=False,
    )
    epochs
    return (epochs,)


@app.cell
def _(epochs, mo):
    mo.md(f"""
    - Number of epochs: **{len(epochs)}**
    - Epoch data shape (epochs, channels, times): **{epochs.get_data().shape}**
    - Sampling frequency: **{epochs.info['sfreq']:.0f} Hz**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Channel selection

    One left and one right gradiometer, chosen automatically as proxies for left/right M1.
    The scatter shows every gradiometer's sensor position (top-down view); the two picks
    are highlighted.
    """)
    return


@app.cell
def _(epochs, select_left_right_motor_like_grads):
    left_ch, right_ch = select_left_right_motor_like_grads(epochs.info, verbose=True)
    (left_ch, right_ch)
    return left_ch, right_ch


@app.cell
def _(epochs, left_ch, mne, np, plt, right_ch):
    _picks = mne.pick_types(epochs.info, meg="grad", eeg=False, eog=False, stim=False)
    _names = [epochs.info["ch_names"][p] for p in _picks]
    _pos = np.array([epochs.info["chs"][p]["loc"][:2] for p in _picks])

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(_pos[:, 0], _pos[:, 1], s=12, c="lightgrey", label="grad sensors")
    for _ch, _colour in [(left_ch, "tab:blue"), (right_ch, "tab:red")]:
        _i = _names.index(_ch)
        _ax.scatter(_pos[_i, 0], _pos[_i, 1], s=80, c=_colour, label=_ch)
        _ax.annotate(_ch, _pos[_i], textcoords="offset points", xytext=(6, 6))
    _ax.set_xlabel("x (left ← → right)")
    _ax.set_ylabel("y (posterior ← → anterior)")
    _ax.set_title("Selected left/right sensors")
    _ax.set_aspect("equal")
    _ax.legend(loc="lower right", fontsize=8)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Two-channel signal

    Restrict to the two selected channels, giving data of shape (n_epochs, 2, n_times).
    Averaging across epochs gives the evoked response for each channel.
    """)
    return


@app.cell
def _(epochs, get_two_channel_data, left_ch, right_ch):
    data_2ch = get_two_channel_data(epochs, left_ch, right_ch)
    data_2ch.shape
    return (data_2ch,)


@app.cell
def _(data_2ch, epochs, left_ch, plt, right_ch):
    _t = epochs.times
    _avg = data_2ch.mean(axis=0)  # (2, n_times)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(_t, _avg[0], label=f"left ({left_ch})", color="tab:blue")
    _ax.plot(_t, _avg[1], label=f"right ({right_ch})", color="tab:red")
    _ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Evoked amplitude (T/m)")
    _ax.set_title("Trial-averaged signal for the two selected channels")
    _ax.legend()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sliding windows

    The metadata driving the per-window connectivity: each row is one 500 ms window,
    stepped by 50 ms across the epoch.
    """)
    return


@app.cell
def _(
    EPOCH_TMAX,
    EPOCH_TMIN,
    STEP_SEC,
    WIN_LEN_SEC,
    data_2ch,
    epochs,
    make_sliding_windows,
    np,
    pd,
):
    _sfreq = epochs.info["sfreq"]
    _n_times = data_2ch.shape[2]

    windows = make_sliding_windows(
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        win_len=WIN_LEN_SEC,
        step=STEP_SEC,
        sfreq=_sfreq,
        n_times=_n_times,
    )
    times = np.array([w["center_sec"] for w in windows])
    windows_df = pd.DataFrame(windows)
    windows_df
    return times, windows


@app.cell
def _(mo):
    mo.md(r"""
    ## Per-window connectivity

    For each window, coherence and wPLI between the two channels across the beta band.
    This is the heavy step; it runs once and its result feeds the reactive views below.
    """)
    return


@app.cell
def _(
    FMAX,
    FMIN,
    METHODS,
    compute_connectivity_per_window,
    data_2ch,
    epochs,
    windows,
):
    raw_results, freqs = compute_connectivity_per_window(
        data_2ch=data_2ch,
        sfreq=epochs.info["sfreq"],
        windows=windows,
        fmin=FMIN,
        fmax=FMAX,
        methods=METHODS,
    )
    print(f"Actual Fourier frequency bins (Hz): {freqs}")
    print(f"Result shape per method (windows, freqs): {raw_results[METHODS[0]].shape}")
    return freqs, raw_results


@app.cell
def _(mo):
    mo.md(r"""
    ## Tidy connectivity table

    The per-window, per-frequency connectivity values in long form: one row per
    (method, time, frequency). This is the underlying table the charts below draw from.
    """)
    return


@app.cell
def _(freqs, np, pd, raw_results, times):
    _rows = []
    for _method, _arr in raw_results.items():
        _tt, _ff = np.meshgrid(times, freqs, indexing="ij")
        _rows.append(
            pd.DataFrame(
                {
                    "method": _method,
                    "time_sec": _tt.ravel(),
                    "freq_hz": _ff.ravel(),
                    "connectivity": _arr.ravel(),
                }
            )
        )
    tidy_df = pd.concat(_rows, ignore_index=True)
    tidy_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Saved outputs

    The same artefacts as the original script (an NPZ, per-method CSVs, and PNG figures)
    written to `somato_fc_results/` at the repo root. Saving is behind a button so opening
    the notebook does not overwrite the committed results on every run. Unlike the script,
    the CSV header uses the actual frequency bins, so it matches the data columns.
    """)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save outputs to somato_fc_results/")
    save_button
    return (save_button,)


@app.cell
def _(
    EPOCH_TMAX,
    EPOCH_TMIN,
    METHODS,
    Path,
    STEP_SEC,
    SUBJECT,
    TASK,
    WIN_LEN_SEC,
    freqs,
    left_ch,
    mo,
    np,
    plt,
    raw_results,
    right_ch,
    save_button,
    times,
):
    mo.stop(not save_button.value, mo.md("Press the button above to write the output files."))

    _base = mo.notebook_dir()
    _out_dir = (_base.parent if _base is not None else Path(".")) / "somato_fc_results"
    _out_dir.mkdir(exist_ok=True, parents=True)

    _meta = dict(
        subject=SUBJECT,
        task=TASK,
        left_ch=left_ch,
        right_ch=right_ch,
        epoch_tmin=EPOCH_TMIN,
        epoch_tmax=EPOCH_TMAX,
        win_len_sec=WIN_LEN_SEC,
        step_sec=STEP_SEC,
        freqs_hz=freqs.tolist(),
    )
    np.savez(
        _out_dir / "somato_interhemispheric_fc.npz",
        times=times,
        freqs=freqs,
        raw_coh=raw_results.get("coh"),
        raw_wpli=raw_results.get("wpli"),
        meta=_meta,
    )

    _written = ["somato_interhemispheric_fc.npz"]
    for _method in METHODS:
        _arr = raw_results[_method]

        _header = "time_sec," + ",".join(f"{f:.1f}Hz" for f in freqs)
        np.savetxt(
            _out_dir / f"raw_{_method}.csv",
            np.column_stack([times, _arr]),
            delimiter=",",
            header=_header,
            comments="",
        )

        _fig, _ax = plt.subplots(figsize=(10, 4))
        _im = _ax.imshow(
            _arr.T,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        _ax.axvline(0.0, color="w", linestyle="--", linewidth=1)
        _ax.set_xlabel("Time (s)")
        _ax.set_ylabel("Frequency (Hz)")
        _ax.set_title(f"Raw {_method} connectivity ({left_ch} - {right_ch})")
        _cbar = plt.colorbar(_im, ax=_ax)
        _cbar.set_label("Connectivity")
        _fig.tight_layout()
        _fig.savefig(_out_dir / f"raw_{_method}_time_frequency.png", dpi=150)
        plt.close(_fig)

        _mean = _arr.mean(axis=1)
        _fig2, _ax2 = plt.subplots(figsize=(10, 4))
        _ax2.plot(times, _mean, linewidth=2)
        _ax2.axvline(0.0, color="k", linestyle="--", linewidth=1)
        _ax2.set_xlabel("Time (s)")
        _ax2.set_ylabel("Mean beta connectivity")
        _ax2.set_title(f"Raw {_method} ({freqs[0]:.0f}-{freqs[-1]:.0f} Hz avg)")
        _fig2.tight_layout()
        _fig2.savefig(_out_dir / f"raw_{_method}_beta_average.png", dpi=150)
        plt.close(_fig2)

        _written += [
            f"raw_{_method}.csv",
            f"raw_{_method}_time_frequency.png",
            f"raw_{_method}_beta_average.png",
        ]

    mo.md(f"Wrote to `{_out_dir}`:\n\n- " + "\n- ".join(_written))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reactive views

    Pick a method and an averaging band. The time-frequency map shows connectivity across
    the full beta band; the line below collapses the chosen band to a single value per
    window (a flat average over frequency), the "raw connectivity" summary from the meeting.
    """)
    return


@app.cell
def _(METHODS, freqs, mo):
    method_selector = mo.ui.dropdown(
        options=METHODS, value=METHODS[0], label="Connectivity method"
    )
    band = mo.ui.range_slider(
        start=int(freqs[0]),
        stop=int(freqs[-1]),
        value=[int(freqs[0]), int(freqs[-1])],
        step=1,
        label="Averaging band (Hz)",
    )
    mo.hstack([method_selector, band], justify="start", gap=2)
    return band, method_selector


@app.cell
def _(freqs, method_selector, plt, raw_results, times):
    _method = method_selector.value
    _arr = raw_results[_method]

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _im = _ax.imshow(
        _arr.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
    )
    _ax.axvline(0.0, color="w", linestyle="--", linewidth=1)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Frequency (Hz)")
    _ax.set_title(f"{_method} connectivity across the beta band")
    _cbar = plt.colorbar(_im, ax=_ax)
    _cbar.set_label("Connectivity")
    _fig
    return


@app.cell
def _(band, freqs, method_selector, np, plt, raw_results, times):
    _method = method_selector.value
    _lo, _hi = band.value
    _mask = (freqs >= _lo) & (freqs <= _hi)
    _mean = raw_results[_method][:, _mask].mean(axis=1)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(times, _mean, linewidth=2, color="tab:purple")
    _ax.axvline(0.0, color="k", linestyle="--", linewidth=1)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Mean connectivity")
    _ax.set_title(
        f"{_method}, averaged over {_lo:.0f}-{_hi:.0f} Hz "
        f"({int(np.count_nonzero(_mask))} freq bins)"
    )
    _fig
    return


if __name__ == "__main__":
    app.run()
