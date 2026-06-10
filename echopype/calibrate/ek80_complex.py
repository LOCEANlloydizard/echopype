from collections import defaultdict
from functools import partial
from typing import Dict, Literal, Optional, Union

import numpy as np
import xarray as xr
from scipy import signal

from ..convert.set_groups_ek80 import DECIMATION, FILTER_IMAG, FILTER_REAL


def _compute_power_from_complex_signal(
    signal: xr.DataArray,
    z_et,
    z_er,
) -> xr.DataArray:
    """Calculate received electrical power from sector-level complex samples.

    The input is expected to retain the ``beam`` dimension. The function
    averages over transducer sectors internally before converting the
    complex signal to received electrical power.

    Equivalent to CRIMAC ``calcPower``.
    """
    prx = (
        signal["beam"].size
        * np.abs(signal.mean(dim="beam")) ** 2
        / (2 * np.sqrt(2)) ** 2
        * (np.abs(z_er + z_et) / np.abs(z_er)) ** 2
        / np.abs(z_et)
    )

    prx = prx.where(prx > 0, 1e-20)
    prx.name = "received_power"

    return prx


def _align_autocorrelation(
    mf_auto: np.ndarray,
    pc_target: np.ndarray,
) -> np.ndarray:
    """Align matched-filter autocorrelation to target echo.

    Equivalent to CRIMAC ``alignAuto``.
    """
    idx_peak_auto = np.argmax(np.abs(mf_auto))
    idx_peak_target = np.argmax(np.abs(pc_target))

    left_samples = idx_peak_target
    right_samples = len(pc_target) - idx_peak_target

    idx_start = max(0, idx_peak_auto - left_samples)
    idx_stop = min(len(mf_auto), idx_peak_auto + right_samples)

    return mf_auto[idx_start:idx_stop]


def _compute_ts_spectrum(
    pc_target: np.ndarray,
    mf_auto_red: np.ndarray,
    NFFT: int,
    frequency: np.ndarray,
    fs_dec: float,
):
    """Compute target, autocorrelation, and normalised DFTs for TS spectrum.

    Equivalent to CRIMAC ``calcDFTforTS``, with explicit NFFT.
    """
    frequency_index = np.mod(
        np.floor(frequency / fs_dec * NFFT).astype(int),
        NFFT,
    )

    pc_target_spectrum = np.fft.fft(pc_target, n=NFFT)[frequency_index]
    mf_auto_red_spectrum = np.fft.fft(mf_auto_red, n=NFFT)[frequency_index]

    normalized_spectrum = pc_target_spectrum / mf_auto_red_spectrum

    return pc_target_spectrum, mf_auto_red_spectrum, normalized_spectrum


def _get_hanning_window(
    sound_speed: float,
    tau: float,
    sample_interval: float,
):
    """Get normalized Hann window for Sv(f) sliding-window FFT.

    Equivalent to CRIMAC ``defHanningWindow``.
    """
    dr = sample_interval * sound_speed / 2
    fs_dec = 1 / sample_interval

    L = sound_speed * 2 * tau / dr
    n_window = int(2 ** np.ceil(np.log2(L)))

    window = np.hanning(n_window)
    window = window / (np.linalg.norm(window) / np.sqrt(n_window))

    window_duration = n_window / fs_dec

    return window, n_window, window_duration, fs_dec


def _compute_svf_spectrum(
    pc_spread: np.ndarray,
    range_meter: np.ndarray,
    window: np.ndarray,
    n_window: int,
    frequency_index: np.ndarray,
    mf_auto_spectrum: np.ndarray,
    step: int = 1,
):
    """Compute normalized sliding-window spectrum for Sv(f).

    Equivalent to CRIMAC ``calcDFTforSv``.
    """
    mf_auto_spectrum_f = mf_auto_spectrum[frequency_index]

    normalized_spectrum = []
    svf_range = []

    for start in range(0, len(pc_spread) - n_window, step):
        stop = start + n_window

        windowed_signal = window * pc_spread[start:stop]
        signal_spectrum = np.fft.fft(windowed_signal, n=n_window)[frequency_index]

        normalized_spectrum.append(signal_spectrum / mf_auto_spectrum_f)

        center = (start + stop) // 2
        svf_range.append(range_meter[center])

    return np.asarray(normalized_spectrum), np.asarray(svf_range)


def _compute_ts_spectrum_calibrated(
    power_spectrum: np.ndarray,
    target_range: float,
    frequency: np.ndarray,
    sound_speed: float,
    absorption_f: np.ndarray,
    transmit_power: float,
    gain_f: np.ndarray,
):
    """Apply CRIMAC-style TS(f) calibration equation.

    Equivalent to CRIMAC ``calcTSf``.
    """
    wavelength_f = sound_speed / frequency

    return (
        10 * np.log10(power_spectrum)
        + 40 * np.log10(target_range)
        + 2 * absorption_f * target_range
        - 10 * np.log10(transmit_power * wavelength_f**2 * gain_f**2 / (16 * np.pi**2))
    )


def _compute_svf_power(
    normalized_spectrum: np.ndarray,
    n_beams: int,
    z_et: float,
    z_er: float,
):
    """Convert normalized Sv(f) spectrum to received power spectrum.

    Equivalent to CRIMAC ``calcPowerFreqSv``.
    """
    return _compute_complex_power(
        normalized_spectrum=normalized_spectrum,
        n_beams=n_beams,
        z_et=z_et,
        z_er=z_er,
    )


def _get_svf_frequency_grid(
    f_start: float,
    f_stop: float,
    frequency_resolution: float,
    fs_dec: float,
    n_window: int,
):
    """Get Sv(f) frequency grid and corresponding FFT indices."""

    n_f_points = int(np.floor((f_stop - f_start) / frequency_resolution)) + 1
    frequency = f_start + np.arange(n_f_points) * frequency_resolution

    frequency_index = np.mod(
        np.floor(frequency / fs_dec * n_window).astype(int),
        n_window,
    )

    return frequency, frequency_index


def _compute_ts_spectrum_power(
    normalized_spectrum: np.ndarray,
    n_beams: int,
    z_et: float,
    z_er: float,
):
    """Convert normalised TS(f) spectrum to received power spectrum.

    Equivalent to CRIMAC ``calcPowerFreqTS``.
    """
    return _compute_complex_power(
        normalized_spectrum=normalized_spectrum,
        n_beams=n_beams,
        z_et=z_et,
        z_er=z_er,
    )


def _get_autocorrelation(
    matched_filter: np.ndarray,
    n_window: int,
):
    """Get matched-filter autocorrelation spectrum.

    Equivalent to CRIMAC ``calcAutoCorrelation``.
    """
    mf_auto = (
        np.convolve(
            matched_filter,
            np.conj(matched_filter[::-1]),
            mode="full",
        )
        / np.linalg.norm(matched_filter) ** 2
    )

    mf_auto_spectrum = np.fft.fft(mf_auto, n=n_window)

    return mf_auto, mf_auto_spectrum


def _get_pulse_compressed_signal(
    beam: xr.Dataset,
    matched_filter: Dict,
) -> xr.DataArray:
    """Calculate pulse-compressed complex samples for each transducer sector.

    Equivalent to CRIMAC ``calcPulseCompressedSignals``.
    """
    pc = compress_pulse(
        backscatter=beam["backscatter_r"] + 1j * beam["backscatter_i"],
        chirp=matched_filter,
    )
    pc = pc / get_norm_fac(chirp=matched_filter)
    pc.name = "pulse_compressed_signal"

    return pc


def _get_average_signal(
    signal: xr.DataArray,
) -> xr.DataArray:
    """Average complex signal over transducer sectors.

    Equivalent to CRIMAC ``calcAverageSignal``.
    """
    out = signal.mean(dim="beam")
    out.name = "average_signal"

    return out


def _compute_complex_power(
    normalized_spectrum: np.ndarray,
    n_beams: int,
    z_et: float,
    z_er: float,
):
    impedance_factor = (np.abs(z_er + z_et) / np.abs(z_er)) ** 2 / np.abs(z_et)

    return n_beams * (np.abs(normalized_spectrum) / (2 * np.sqrt(2))) ** 2 * impedance_factor


def tapered_chirp(
    fs,
    transmit_duration_nominal,
    slope,
    transmit_frequency_start,
    transmit_frequency_stop,
    drop_last_hanning_zero=False,
):
    """
    Create the chirp replica following implementation from Lars Anderson.

    Ref source: https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-Raw-To-Svf-TSf/blob/main/Core/Calculation.py
    """  # noqa: E501

    tau = transmit_duration_nominal
    f0 = transmit_frequency_start
    f1 = transmit_frequency_stop

    nsamples = int(np.floor(tau * np.float32(fs))[0])
    t = np.linspace(0, nsamples - 1, num=nsamples) * 1 / fs
    a = np.pi * (f1 - f0) / tau
    b = 2 * np.pi * f0
    y = np.cos(a * t * t + b * t)
    L = int(np.round(tau * fs * slope * 2.0)[0])  # Length of hanning window
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(0, L, 1) / (L - 1)))
    N = len(y)
    w1 = w[0 : int(len(w) / 2)]
    if drop_last_hanning_zero:
        # pyEcholab implementation drops the last value in the hanning window (which is zero)
        w2 = w[int(len(w) / 2) : -1]
    else:
        # CRIMAC implementation keeps the last value
        w2 = w[int(len(w) / 2) :]
    i0 = 0
    i1 = len(w1)
    i2 = N - len(w2)
    i3 = N
    y[i0:i1] = y[i0:i1] * w1
    y[i2:i3] = y[i2:i3] * w2

    return y / np.max(y), t  # amplitude needs to be normalized


def filter_decimate_chirp(coeff_ch: Dict, y_ch: np.array, fs: float):
    """Filter and decimate the transmit replica for one channel.

    Parameters
    ----------
    coeff_ch : dict
        a dictionary containing filter coefficients and decimation factors for ``ch_id``
    y_ch : np.array
        chirp from _tapered_chirp
    fs : float
        system sampling frequency [Hz]
    """
    # Get values

    # WBT filter and decimation
    ytx_wbt = signal.convolve(y_ch, coeff_ch["wbt_fil"])
    ytx_wbt_deci = ytx_wbt[0 :: int(coeff_ch["wbt_decifac"])]

    # PC filter and decimation
    ytx_pc = signal.convolve(ytx_wbt_deci, coeff_ch["pc_fil"])
    ytx_pc_deci = ytx_pc[0 :: int(coeff_ch["pc_decifac"])]
    ytx_pc_deci_time = (
        np.arange(ytx_pc_deci.size) * 1 / fs * coeff_ch["wbt_decifac"] * coeff_ch["pc_decifac"]
    )

    return ytx_pc_deci, ytx_pc_deci_time


def get_vend_filter_EK80(
    vend: xr.Dataset,
    channel_id: str,
    filter_name: Literal["WBT", "PC"],
    param_type: Literal["coeff", "decimation"],
) -> Optional[Union[np.ndarray, int]]:
    """
    Get filter coefficients stored in the Vendor_specific group attributes.

    Parameters
    ----------
    vend: xr.Dataset
        An xr.Dataset from EchoData["Vendor_specific"]
    channel_id : str
        channel id for which the param to be retrieved
    filter_name : str
        name of filter coefficients to retrieve
    param_type : str
        'coeff' or 'decimation'

    Returns
    -------
    np.ndarray or int or None
        The filter coefficient or the decimation factor
    """
    var_imag = f"{filter_name}_{FILTER_IMAG}"
    var_real = f"{filter_name}_{FILTER_REAL}"
    var_df = f"{filter_name}_{DECIMATION}"

    # if the variables are not in the dataset, simply return None
    if not all([var in vend for var in [var_imag, var_real, var_df]]):
        return None

    # Select the channel requested
    sel_vend = vend.sel(channel=channel_id)

    if param_type == "coeff":
        # Compute complex number from imaginary and real parts
        v_complex = sel_vend[var_real] + 1j * sel_vend[var_imag]
        # Drop nan fillers and get the values
        v = v_complex.dropna(dim=f"{filter_name}_filter_n").values
        return v
    else:
        # Get the decimation value
        return sel_vend[var_df].values


def get_filter_coeff(vend: xr.Dataset) -> Dict:
    """
    Get WBT and PC filter coefficients for constructing the transmit replica.

    Parameters
    ----------
    vend: xr.Dataset
        An xr.Dataset from EchoData["Vendor_specific"]

    Returns
    -------
    dict
        A dictionary indexed by ``channel`` and values being dictionaries containing
        filter coefficients and decimation factors for constructing the transmit replica.
    """
    # Select first index of filter time, which is of length 1. This ensures that the
    # coefficient and decimation arrays are of shape (n,) instead of shape (1, n,).
    # TODO: is n here the number of channels?
    if "filter_time" in vend.dims:
        vend = vend.isel(filter_time=0)

    coeff = defaultdict(dict)
    for ch_id in vend["channel"].values:
        # filter coefficients and decimation factor
        coeff[ch_id]["wbt_fil"] = get_vend_filter_EK80(vend, ch_id, "WBT", "coeff")
        coeff[ch_id]["pc_fil"] = get_vend_filter_EK80(vend, ch_id, "PC", "coeff")
        coeff[ch_id]["wbt_decifac"] = get_vend_filter_EK80(vend, ch_id, "WBT", "decimation")
        coeff[ch_id]["pc_decifac"] = get_vend_filter_EK80(vend, ch_id, "PC", "decimation")

    return coeff


def get_tau_effective(
    ytx_dict: Dict[str, np.array],
    fs_deci_dict: Dict[str, float],
    waveform_mode: str,
    channel: xr.DataArray,
    ping_time: xr.DataArray,
):
    """Compute effective pulse length.

    Parameters
    ----------
    ytx_dict : dict
        A dict of transmit signals, with keys being the ``channel`` and
        values being either a vector when the transmit signals are identical across all pings
        or a 2D array when the transmit signals vary across ping
    fs_deci_dict : dict
        A dict of sampling frequency of the decimated (recorded) signal,
        with keys being the ``channel``
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    """
    tau_effective = {}
    for ch, ytx in ytx_dict.items():
        if waveform_mode == "BB":
            ytxa = signal.convolve(ytx, np.flip(np.conj(ytx))) / np.linalg.norm(ytx) ** 2
            ptxa = np.abs(ytxa) ** 2
        elif waveform_mode == "CW":
            ptxa = np.abs(ytx) ** 2  # energy of transmit signal
        tau_effective[ch] = ptxa.sum() / (ptxa.max() * fs_deci_dict[ch])

    # set up coordinates
    if len(ytx.shape) == 1:  # ytx is a vector (transmit signals are identical across pings)
        coords = {"channel": channel}
    elif len(ytx.shape) == 2:  # ytx is a matrix (transmit signals vary across pings)
        coords = {"channel": channel, "ping_time": ping_time}

    vals = np.array(list(tau_effective.values())).squeeze()
    if vals.size == 1:
        vals = np.expand_dims(vals, axis=0)

    tau_effective = xr.DataArray(
        data=vals,
        coords=coords,
    )

    return tau_effective


def get_transmit_signal(
    beam: xr.Dataset,
    coeff: Dict,
    waveform_mode: str,
    fs: Union[float, xr.DataArray],
    drop_last_hanning_zero: bool = False,
):
    """Reconstruct transmit signal and compute effective pulse length.

    Parameters
    ----------
    beam : xr.Dataset
        EchoData["Sonar/Beam_group1"] selected with channel subset
    coeff : dict
        a dictionary indexed by ``channel`` and values being dictionaries containing
        filter coefficients and decimation factors for constructing the transmit replica.
    waveform_mode : str
        ``CW`` for CW-mode samples, either recorded as complex or power samples
        ``BB`` for BB-mode samples, recorded as complex samples
    drop_last_hanning_zero: bool, default False
        If true, uses the pyEcholab implementation of dropping the hanning window's
        last index value (which is zero). Else, follows the CRIMAC implementation and
        keeps the last zero. This is here for CI test purposes.

    Return
    ------
    y_all
        Transmit replica (BB: broadband chirp, CW: constant frequency sinusoid)
    y_time_all
        Timestamp for the transmit replica
    """
    # Make sure it is BB mode data
    # This is already checked in calibrate_ek
    # but keeping this here for use as standalone function
    if waveform_mode == "BB" and np.all(beam["transmit_type"] == "CW"):
        raise TypeError("File does not contain BB mode complex samples!")
    # Generate all transmit replica
    y_all = {}
    y_time_all = {}
    tx_param_names = [
        "transmit_duration_nominal",
        "slope",
        "transmit_frequency_start",
        "transmit_frequency_stop",
    ]
    for ch in beam["channel"].values:
        tx_params = {}
        fs_chan = fs.sel(channel=ch).data if isinstance(fs, xr.DataArray) else fs
        for p in tx_param_names:
            if waveform_mode == "CW" and p in [
                "transmit_frequency_start",
                "transmit_frequency_stop",
            ]:
                tx_params[p] = np.unique(beam["frequency_nominal"].sel(channel=ch).values)
            else:
                # Extract beam values and filter out NaNs
                beam_values = np.unique(beam[p].sel(channel=ch))
                # Filter out NaN values
                beam_values_without_nan = beam_values[~np.isnan(beam_values)]
                tx_params[p] = beam_values_without_nan
            if tx_params[p].size != 1:
                raise TypeError("File contains changing %s!" % p)
        tx_params["fs"] = fs_chan
        tx_params["drop_last_hanning_zero"] = drop_last_hanning_zero
        y_ch, _ = tapered_chirp(**tx_params)
        # Filter and decimate chirp template
        y_ch, y_tmp_time = filter_decimate_chirp(coeff_ch=coeff[ch], y_ch=y_ch, fs=fs_chan)
        # Fill into output dict
        y_all[ch] = y_ch
        y_time_all[ch] = y_tmp_time

    return y_all, y_time_all


def _convolve_per_channel(backscatter_subset: np.ndarray, replica_dict: dict, channels: dict):
    """
    Convolve `backscatter_subset` array along range sample dimension for each channel.
    The `backscatter_subset` array is a numpy array and has implicit dimensions
    `('range_sample', 'channel')`.

    When the `backscatter_subset` array is all 0s, we return it since the resulting
    convolution will be all 0s, irrespective of what the corresponding transmit
    signal is.

    When this function is used in `compress_pulse`, the array that is being sent
    as backscatter subset corresponds to a specific `ping_time` and `beam`, from
    the backscatter array.
    """
    # Return if all 0s
    if np.all(backscatter_subset == 0.0 + 0.0j):
        return backscatter_subset
    else:
        # Create zeros like array from `backscatter_subset`
        convolved = np.zeros_like(backscatter_subset, dtype=np.complex64)
        # Iterate over channels
        for ch_seq, channel in enumerate(channels):
            # Extract replica values
            replica = replica_dict[str(channel.values)]
            # Convolve backscatter and chirp replica
            convolved[:, ch_seq] = signal.convolve(
                backscatter_subset[:, ch_seq], replica, mode="full"
            )[replica.size - 1 :]
        return convolved


def compress_pulse(backscatter: xr.DataArray, chirp: Dict) -> xr.DataArray:
    """Perform pulse compression on the backscatter data.

    Parameters
    ----------
    backscatter : xr.DataArray
        complex backscatter samples
    chirp : dict
        transmit chirp replica indexed by ``channel``

    Returns
    -------
    xr.DataArray
        A data array containing pulse compression output.
    """
    # Calculate the transmit signal values from the chirp dictionary
    replica_dict = {
        # Compute conjugate and flip for each channel's transmit signal
        str(channel.values): np.flipud(np.conj(chirp[str(channel.values)]))
        for channel in backscatter["channel"]
    }

    # Zero out backscatter NaN values
    nan_mask = np.isnan(backscatter)
    backscatter_with_zeroed_nans = xr.where(nan_mask, 0.0 + 0.0j, backscatter)

    # Create a partial function of the convolve function to pass in chirp and channels
    _convolve_per_channel_partial = partial(
        _convolve_per_channel,
        replica_dict=replica_dict,
        channels=backscatter_with_zeroed_nans["channel"],
    )

    # Apply convolve on backscatter and replica (along range sample and channel dimension):
    # To enable parallelized computation with `dask='parallelized'`, we rechunk to ensure that
    #  the data is chunked with only one chunk along the core dimensions.
    if backscatter_with_zeroed_nans.chunks is not None:
        backscatter_with_zeroed_nans = backscatter_with_zeroed_nans.chunk(
            {"range_sample": -1, "channel": -1}
        )
    pc = xr.apply_ufunc(
        _convolve_per_channel_partial,
        backscatter_with_zeroed_nans,
        input_core_dims=[["range_sample", "channel"]],
        output_core_dims=[["range_sample", "channel"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.complex64],
    )

    # Restore NaN values in the pulse compressed array
    pc = xr.where(nan_mask, np.nan, pc)

    return pc


def get_norm_fac(chirp: Dict) -> xr.DataArray:
    """
    Get normalization factor from the chirp dictionary.

    Parameters
    ----------
    chirp : dict
        transmit chirp replica indexed by ``channel``

    Returns
    -------
    xr.DataArray
        A data array containing the normalization factor, with channel coordinate
    """
    norm_fac = []
    ch_all = []
    for ch, tx in chirp.items():
        norm_fac.append(np.linalg.norm(tx) ** 2)
        ch_all.append(ch)
    return xr.DataArray(norm_fac, coords={"channel": ch_all})
