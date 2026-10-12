#!/usr/bin/env python3
"""Transparent, provenance-friendly spectral connectivity across epochs.

This module deliberately implements a small, inspectable subset of
``mne_connectivity.spectral_connectivity_epochs``.  It is designed for the
MNE somato/Marimo/Fluid demonstration where every important intermediate
array should remain available for linked visualisation.

Implemented
-----------
* Input: NumPy arrays shaped (epochs, signals, times), ``mne.Epochs``, or an
  iterable of arrays shaped (signals, times).
* Spectrum mode: ``mode='fourier'`` using a Hann window and one-sided FFT.
* Bivariate methods: ``coh``, ``cohy``, ``imcoh``, ``plv``, ``ciplv``,
  ``ppc``, ``pli``, ``dpli``, ``wpli``, and ``wpli2_debiased``.
* Explicit connection indices or MNE-style lower-triangular all-to-all output.
* Frequency selection, frequency decimation, multiple frequency bands, and
  optional averaging within bands.
* A result object with ``get_data()``, ``freqs``, ``names``, ``indices``, and
  a ``provenance`` dictionary containing the intermediate calculations.

Not implemented
---------------
Multitaper, Morlet CWT, precomputed Spectrum/TFR input, multivariate methods,
Granger causality, parallel execution, or block-wise processing.  Parameters
for these features are retained in the public signature so an existing call
can import this function with minimal changes, but unsupported requests fail
clearly rather than silently producing different science.

This is an independent educational implementation, not a copy of MNE-
Connectivity source code and not a drop-in replacement for every use case.
Validate it against the version of MNE-Connectivity used in a study before
using it for inferential analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import warnings

import numpy as np


_EPS = np.finfo(float).eps
_SUPPORTED_METHODS = {
    "coh", "cohy", "imcoh", "plv", "ciplv", "ppc", "pli", "dpli",
    "wpli", "wpli2_debiased",
}


@dataclass
class TransparentSpectralConnectivity:
    """Small MNE-like connectivity container with inspectable provenance."""

    data: np.ndarray
    freqs: np.ndarray
    names: list[Any]
    method: str
    n_nodes: int
    indices: tuple[np.ndarray, np.ndarray] | None
    n_epochs_used: int
    times_used: np.ndarray
    freqs_used: Any
    spec_method: str
    provenance: dict[str, Any]

    def get_data(self, output: str = "raveled") -> np.ndarray:
        """Return connectivity values.

        Parameters
        ----------
        output : {'raveled', 'dense'}
            ``'raveled'`` returns the same stored shape used by ordinary MNE
            calls. ``'dense'`` converts explicitly indexed connections into a
            dense ``(nodes, nodes, frequencies)`` representation.
        """
        if output == "raveled":
            return self.data
        if output != "dense":
            raise ValueError("output must be 'raveled' or 'dense'")
        if self.indices is None:
            return self.data.reshape((self.n_nodes, self.n_nodes) + self.data.shape[1:])
        dense = np.zeros((self.n_nodes, self.n_nodes) + self.data.shape[1:], dtype=self.data.dtype)
        dense[self.indices] = self.data
        return dense

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.data, dtype=dtype)


# ---------------------------------------------------------------------------
# Small computational functions.  Each returns an independently inspectable
# result suitable for a provenance-linked visualisation.
# ---------------------------------------------------------------------------

def coerce_epoched_data(data: Any, sfreq: float | None, names=None):
    """Convert supported input to ``(epochs, signals, times)`` NumPy data."""
    input_times = None
    metadata = {}

    if hasattr(data, "get_data") and hasattr(data, "info"):
        array = np.asarray(data.get_data())
        sfreq_in = float(data.info["sfreq"])
        input_times = np.asarray(data.times, dtype=float)
        names_in = list(data.ch_names)
        metadata = {
            "events": getattr(data, "events", None),
            "event_id": getattr(data, "event_id", None),
            "metadata": getattr(data, "metadata", None),
        }
        sfreq = sfreq_in
        if names is None:
            names = names_in
    elif isinstance(data, np.ndarray):
        array = np.asarray(data)
    else:
        array = np.asarray(list(data))

    if array.ndim != 3:
        raise ValueError("data must have shape (n_epochs, n_signals, n_times)")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("data must be numeric")
    if np.iscomplexobj(array):
        raise NotImplementedError("Precomputed complex spectra/TFR input is not implemented")
    if sfreq is None:
        raise ValueError("Sampling frequency (sfreq) is required with array input")
    if float(sfreq) <= 0:
        raise ValueError("sfreq must be positive")

    n_epochs, n_signals, n_times = array.shape
    if n_epochs < 1 or n_signals < 1 or n_times < 2:
        raise ValueError("data must contain at least 1 epoch, 1 signal, and 2 samples")
    if names is None:
        names = list(range(n_signals))
    else:
        names = list(names)
        if len(names) != n_signals:
            raise ValueError("names must have one entry per signal")
    if input_times is None:
        input_times = np.arange(n_times, dtype=float) / float(sfreq)

    return array.astype(float, copy=False), float(sfreq), names, input_times, metadata


def select_time_samples(data, times, tmin=None, tmax=None):
    """Select an inclusive time interval and expose its sample indices."""
    lo = times[0] if tmin is None else float(tmin)
    hi = times[-1] if tmax is None else float(tmax)
    if lo > hi:
        raise ValueError("tmin must be <= tmax")
    mask = (times >= lo - 1e-12) & (times <= hi + 1e-12)
    sample_indices = np.flatnonzero(mask)
    if sample_indices.size < 2:
        raise ValueError("The selected time interval must contain at least 2 samples")
    return data[..., mask], times[mask], sample_indices


def make_hann_window(n_times: int) -> np.ndarray:
    """Return the symmetric Hann window used by NumPy's ``hanning``."""
    if n_times < 2:
        raise ValueError("n_times must be >= 2")
    return np.hanning(n_times)


def apply_window(data: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Multiply each signal and epoch by the same spectral window."""
    if data.shape[-1] != window.size:
        raise ValueError("window length must match the time dimension")
    return data * window[np.newaxis, np.newaxis, :]


def compute_one_sided_fourier(windowed_data: np.ndarray, sfreq: float):
    """Compute one-sided complex FFT coefficients and their frequency bins."""
    coefficients = np.fft.rfft(windowed_data, axis=-1)
    freqs = np.fft.rfftfreq(windowed_data.shape[-1], d=1.0 / sfreq)
    return coefficients, freqs


def normalise_frequency_bands(fmin, fmax, epoch_duration):
    """Convert scalar/tuple band limits into aligned one-dimensional arrays."""
    if fmin is None:
        fmin = 5.0 / epoch_duration
    fmins = np.atleast_1d(np.asarray(fmin, dtype=float))
    fmaxs = np.atleast_1d(np.asarray(fmax, dtype=float))
    if fmins.size != fmaxs.size:
        raise ValueError("fmin and fmax must have the same length")
    if np.any(fmins > fmaxs):
        raise ValueError("fmax must be larger than or equal to fmin")
    return fmins, fmaxs


def select_frequency_bins(freqs, fmins, fmaxs, fdecim=1, fskip=0):
    """Select bins in any requested band, then apply frequency decimation."""
    if int(fdecim) != fdecim or fdecim < 1:
        raise ValueError("fdecim must be an integer >= 1")
    if int(fskip) != fskip or fskip < 0:
        raise ValueError("fskip must be an integer >= 0")
    if fskip and fdecim != 1:
        raise ValueError("fskip and fdecim cannot be used together")

    band_masks = [(freqs >= lo) & (freqs <= hi) for lo, hi in zip(fmins, fmaxs)]
    union = np.logical_or.reduce(band_masks)
    candidate = np.flatnonzero(union)
    step = (fskip + 1) if fskip else fdecim
    selected = candidate[::step]
    if selected.size == 0:
        raise ValueError("No Fourier bins fall inside the requested frequency range")
    selected_band_indices = [np.flatnonzero((freqs[selected] >= lo) & (freqs[selected] <= hi))
                             for lo, hi in zip(fmins, fmaxs)]
    if any(idx.size == 0 for idx in selected_band_indices):
        raise ValueError("At least one requested frequency band contains no selected bins")
    return selected, selected_band_indices


def prepare_connection_indices(n_signals: int, indices):
    """Create explicit seed/target arrays and MNE-style storage information."""
    if indices is None:
        seeds, targets = np.tril_indices(n_signals, k=-1)
        return seeds.astype(int), targets.astype(int), None

    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError("indices must be a tuple: (seeds, targets)")
    seeds = np.asarray(indices[0], dtype=int).ravel()
    targets = np.asarray(indices[1], dtype=int).ravel()
    if seeds.size != targets.size:
        raise ValueError("seed and target index arrays must have equal length")
    if seeds.size == 0:
        raise ValueError("indices must define at least one connection")
    if np.any(seeds < 0) or np.any(seeds >= n_signals) or np.any(targets < 0) or np.any(targets >= n_signals):
        raise IndexError("connection indices are outside the signal range")
    return seeds, targets, (seeds.copy(), targets.copy())


def compute_epoch_spectra(fourier_coefficients, seeds, targets):
    """Compute per-epoch auto-spectra, cross-spectra, and unit cross-spectra."""
    x = fourier_coefficients[:, seeds, :]
    y = fourier_coefficients[:, targets, :]

    # Sxy follows the formula used in the public MNE documentation.
    sxy = x * np.conjugate(y)
    sxx = np.real(x * np.conjugate(x))
    syy = np.real(y * np.conjugate(y))
    magnitude = np.abs(sxy)
    unit_sxy = np.divide(sxy, magnitude, out=np.zeros_like(sxy), where=magnitude > _EPS)
    return {
        "x_fourier": x,
        "y_fourier": y,
        "sxy_per_epoch": sxy,
        "sxx_per_epoch": sxx,
        "syy_per_epoch": syy,
        "unit_sxy_per_epoch": unit_sxy,
    }


def average_epoch_spectra(epoch_spectra):
    """Average spectral quantities across independent epochs."""
    return {
        "mean_sxy": epoch_spectra["sxy_per_epoch"].mean(axis=0),
        "mean_sxx": epoch_spectra["sxx_per_epoch"].mean(axis=0),
        "mean_syy": epoch_spectra["syy_per_epoch"].mean(axis=0),
        "mean_unit_sxy": epoch_spectra["unit_sxy_per_epoch"].mean(axis=0),
        "mean_abs_imag_sxy": np.abs(epoch_spectra["sxy_per_epoch"].imag).mean(axis=0),
    }


def compute_connectivity_measure(method, epoch_spectra, averaged_spectra):
    """Compute one bivariate connectivity measure from transparent spectra."""
    method = str(method).lower()
    if method not in _SUPPORTED_METHODS:
        raise NotImplementedError(f"Method {method!r} is not implemented")

    mean_sxy = averaged_spectra["mean_sxy"]
    mean_sxx = averaged_spectra["mean_sxx"]
    mean_syy = averaged_spectra["mean_syy"]
    denom = np.sqrt(mean_sxx * mean_syy)
    coherency = np.divide(mean_sxy, denom, out=np.zeros_like(mean_sxy), where=denom > _EPS)

    if method == "coh":
        return np.abs(coherency)
    if method == "cohy":
        return coherency
    if method == "imcoh":
        return coherency.imag

    unit = epoch_spectra["unit_sxy_per_epoch"]
    mean_unit = averaged_spectra["mean_unit_sxy"]
    n_epochs = unit.shape[0]
    if method == "plv":
        return np.abs(mean_unit)
    if method == "ciplv":
        num = np.abs(mean_unit.imag)
        ciden = np.sqrt(np.maximum(0.0, 1.0 - mean_unit.real ** 2))
        return np.divide(num, ciden, out=np.zeros_like(num), where=ciden > _EPS)
    if method == "ppc":
        if n_epochs < 2:
            return np.full(mean_unit.shape, np.nan)
        # Unbiased estimator of squared PLV.
        return (n_epochs * np.abs(mean_unit) ** 2 - 1.0) / (n_epochs - 1.0)

    imag = epoch_spectra["sxy_per_epoch"].imag
    if method == "pli":
        return np.abs(np.sign(imag).mean(axis=0))
    if method == "dpli":
        # H(0)=0.5 follows the common DPLI convention.
        return (np.mean(imag > 0, axis=0) + 0.5 * np.mean(imag == 0, axis=0))
    if method == "wpli":
        num = np.abs(imag.mean(axis=0))
        wden = np.abs(imag).mean(axis=0)
        return np.divide(num, wden, out=np.zeros_like(num), where=wden > _EPS)
    if method == "wpli2_debiased":
        sum_im = imag.sum(axis=0)
        sum_abs = np.abs(imag).sum(axis=0)
        sum_sq = (imag ** 2).sum(axis=0)
        num = sum_im ** 2 - sum_sq
        wden = sum_abs ** 2 - sum_sq
        return np.divide(num, wden, out=np.zeros_like(num), where=np.abs(wden) > _EPS)

    raise RuntimeError("Unreachable method dispatch")


def average_frequency_bands(values, selected_freqs, selected_band_indices):
    """Average connectivity values within each requested frequency band."""
    averaged = np.stack([values[:, idx].mean(axis=1) for idx in selected_band_indices], axis=1)
    band_freqs = np.asarray([selected_freqs[idx].mean() for idx in selected_band_indices])
    frequencies_used = [selected_freqs[idx].copy() for idx in selected_band_indices]
    return averaged, band_freqs, frequencies_used


def assemble_output(values, n_signals, seeds, targets, explicit_indices):
    """Match MNE's raveled all-to-all storage when ``indices=None``."""
    if explicit_indices is not None:
        return values
    dense = np.zeros((n_signals, n_signals) + values.shape[1:], dtype=values.dtype)
    dense[seeds, targets] = values
    return dense.reshape((n_signals ** 2,) + values.shape[1:])


# ---------------------------------------------------------------------------
# Public MNE-compatible entry point
# ---------------------------------------------------------------------------

def spectral_connectivity_epochs(
    data,
    names=None,
    method="coh",
    indices=None,
    sfreq=None,
    *,
    mode="multitaper",
    fmin=None,
    fmax=np.inf,
    fskip=None,
    fdecim=None,
    faverage=False,
    tmin=None,
    tmax=None,
    mt_bandwidth=None,
    mt_adaptive=False,
    mt_low_bias=True,
    cwt_freqs=None,
    cwt_n_cycles=7,
    gc_n_lags=40,
    rank=None,
    n_components=1,
    block_size=1000,
    n_jobs=1,
    verbose=None,
):
    """Compute transparent Fourier connectivity across independent epochs.

    The signature intentionally mirrors MNE-Connectivity.  This compact
    implementation supports ``mode='fourier'`` and the bivariate methods
    listed in ``_SUPPORTED_METHODS``.  The returned object exposes all major
    intermediate arrays under ``result.provenance``.
    """
    del mt_bandwidth, mt_adaptive, mt_low_bias, cwt_freqs, cwt_n_cycles
    del gc_n_lags, rank, n_components, block_size, verbose

    if mode != "fourier":
        raise NotImplementedError(
            "This transparent implementation currently supports only mode='fourier'. "
            "Pass mode='fourier' as in the somato Marimo notebook."
        )
    if n_jobs != 1:
        raise NotImplementedError("Parallel execution is intentionally not implemented")

    if fskip is not None:
        warnings.warn("fskip is deprecated; prefer fdecim", FutureWarning, stacklevel=2)
        if fdecim is not None:
            raise ValueError("fskip and fdecim cannot be used together")
    fskip_value = 0 if fskip is None else int(fskip)
    fdecim_value = 1 if fdecim is None else int(fdecim)

    methods = [method] if isinstance(method, str) else list(method)
    if not methods:
        raise ValueError("method must contain at least one connectivity measure")
    unsupported = [m for m in methods if m not in _SUPPORTED_METHODS]
    if unsupported:
        raise NotImplementedError(f"Unsupported method(s): {unsupported}")

    array, sfreq, names, input_times, input_metadata = coerce_epoched_data(data, sfreq, names)
    selected_data, times_used, sample_indices = select_time_samples(array, input_times, tmin, tmax)
    n_epochs, n_signals, n_times = selected_data.shape
    if n_epochs < 2:
        warnings.warn(
            "Connectivity estimated from one epoch is generally unreliable; "
            "spectral_connectivity_epochs averages spectral quantities across epochs.",
            RuntimeWarning,
            stacklevel=2,
        )

    window = make_hann_window(n_times)
    windowed_data = apply_window(selected_data, window)
    all_fourier, all_freqs = compute_one_sided_fourier(windowed_data, sfreq)

    epoch_duration = n_times / sfreq
    fmins, fmaxs = normalise_frequency_bands(fmin, fmax, epoch_duration)
    selected_bins, band_indices = select_frequency_bins(
        all_freqs, fmins, fmaxs, fdecim=fdecim_value, fskip=fskip_value
    )
    selected_freqs = all_freqs[selected_bins]
    fourier = all_fourier[..., selected_bins]

    seeds, targets, explicit_indices = prepare_connection_indices(n_signals, indices)
    epoch_spectra = compute_epoch_spectra(fourier, seeds, targets)
    averaged_spectra = average_epoch_spectra(epoch_spectra)

    provenance_common = {
        "input_data": array,
        "input_times": input_times,
        "selected_sample_indices": sample_indices,
        "selected_data": selected_data,
        "times_used": times_used,
        "hann_window": window,
        "windowed_data": windowed_data,
        "all_fourier_coefficients": all_fourier,
        "all_fourier_frequencies": all_freqs,
        "selected_frequency_indices": selected_bins,
        "selected_frequencies": selected_freqs,
        "fourier_coefficients": fourier,
        "seed_indices": seeds,
        "target_indices": targets,
        **epoch_spectra,
        **averaged_spectra,
        "input_metadata": input_metadata,
        "formula_notes": {
            "sxy": "X * conjugate(Y)",
            "coh": "abs(mean(Sxy)) / sqrt(mean(Sxx) * mean(Syy))",
            "wpli": "abs(mean(imag(Sxy))) / mean(abs(imag(Sxy)))",
        },
    }

    results = []
    for this_method in methods:
        values = compute_connectivity_measure(this_method, epoch_spectra, averaged_spectra)
        freqs_out = selected_freqs
        freqs_used = selected_freqs.copy()
        method_provenance = dict(provenance_common)
        method_provenance["unassembled_connectivity"] = values

        if faverage:
            values, freqs_out, freqs_used = average_frequency_bands(
                values, selected_freqs, band_indices
            )
            method_provenance["frequency_band_indices"] = band_indices
            method_provenance["band_averaged_connectivity"] = values

        output = assemble_output(values, n_signals, seeds, targets, explicit_indices)
        method_provenance["connectivity_output"] = output

        results.append(
            TransparentSpectralConnectivity(
                data=output,
                freqs=np.asarray(freqs_out),
                names=names,
                method=this_method,
                n_nodes=n_signals,
                indices=explicit_indices,
                n_epochs_used=n_epochs,
                times_used=times_used,
                freqs_used=freqs_used,
                spec_method="fourier",
                provenance=method_provenance,
            )
        )

    return results[0] if len(results) == 1 else results


__all__ = [
    "TransparentSpectralConnectivity",
    "spectral_connectivity_epochs",
    "coerce_epoched_data",
    "select_time_samples",
    "make_hann_window",
    "apply_window",
    "compute_one_sided_fourier",
    "normalise_frequency_bands",
    "select_frequency_bins",
    "prepare_connection_indices",
    "compute_epoch_spectra",
    "average_epoch_spectra",
    "compute_connectivity_measure",
    "average_frequency_bands",
    "assemble_output",
]


if __name__ == "__main__":
    # Small deterministic smoke test, independent of MNE/MNE-Connectivity.
    rng = np.random.default_rng(7)
    sfreq = 200.0
    times = np.arange(100) / sfreq
    shared = np.sin(2 * np.pi * 20 * times)
    x = shared + 0.2 * rng.standard_normal((20, 100))
    y = np.sin(2 * np.pi * 20 * times + np.pi / 5) + 0.2 * rng.standard_normal((20, 100))
    demo = np.stack([x, y], axis=1)

    con = spectral_connectivity_epochs(
        demo,
        method=["coh", "wpli"],
        indices=(np.array([0]), np.array([1])),
        sfreq=sfreq,
        mode="fourier",
        fmin=13,
        fmax=30,
    )
    for result in con:
        print(result.method, result.get_data().shape, result.freqs)
