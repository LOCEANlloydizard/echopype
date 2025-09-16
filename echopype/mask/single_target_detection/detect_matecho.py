# matecho.py
import math

import numpy as np
import xarray as xr

MATECHO_DEFAULTS = {
    "SoundSpeed": 1500.0,
    "TS_threshold": -50.0,  # dB
    "MaxAngleOneWayCompression": 6.0,  # dB (one-way; test uses 2x)
    "MinEchoLength": 0.8,  # in pulse lengths
    "MaxEchoLength": 1.8,
    "MinEchoSpace": 1.0,  # in pulse lengths
    "MinEchoDepthM": 3.0,
    "MaxEchoDepthM": 38.0,
    "tvg_start_sample": 3,  # EK60=3, EK80=1
    "block_len": 1e7 / 3,
    # Beam pattern (fallbacks if no metadata):
    "beamwidth_along_3dB_rad": np.deg2rad(7.0),
    "beamwidth_athwart_3dB_rad": np.deg2rad(7.0),
    "steer_along_rad": 0.0,
    "steer_athwart_rad": 0.0,
    # Sv->TS constant terms; keep 0 if unknown:
    "psi_two_way": 0.0,
    "Sa_correction": 0.0,
    "Sa_EK80_nominal": 0.0,
}


def _fallback_1d_aligned(ds, names, ping_time):
    for nm in names:
        if nm in ds:
            da = ds[nm]
            for tdim in ("ping_time", "time", "time1", "time2"):
                if tdim in da.dims:
                    if tdim != "ping_time":
                        da = da.rename({tdim: "ping_time"})
                    da = da.reindex(
                        ping_time=ping_time, method="nearest", tolerance=np.timedelta64(500, "ms")
                    )
                    return np.asarray(da.values, dtype=float)
            if da.ndim == 1 and da.size == ping_time.size:
                return np.asarray(da.values, dtype=float)
    return None


def detect_matecho(ds: xr.Dataset, params: dict):
    """
    Matecho-inspired single-target detector (CW path).
    Returns the same dict layout as your init_st_struct().
    """
    if params is None:
        raise ValueError("params is required.")
    channel = params.get("channel")
    if channel is None:
        raise ValueError("params['channel'] is required.")
    bottom_da = params.get("bottom_da", None)

    # merge defaults
    p = {
        **MATECHO_DEFAULTS,
        **{k: v for k, v in params.items() if k not in ("channel", "bottom_da")},
    }
    c = float(p["SoundSpeed"])
    TS_thr = float(p["TS_threshold"])

    # --- Select variables
    Sv = ds["Sv"].sel(channel=channel).transpose("ping_time", "range_sample")
    Depth = ds["depth"].sel(channel=channel).transpose("ping_time", "range_sample")
    pt = Sv["ping_time"]

    along = ds.get("angle_alongship")
    if along is not None:
        if "channel" in along.dims:
            along = along.sel(channel=channel)
        along = along.transpose("ping_time", "range_sample")
    athwt = ds.get("angle_athwartship")
    if athwt is not None:
        if "channel" in athwt.dims:
            athwt = athwt.sel(channel=channel)
        athwt = athwt.transpose("ping_time", "range_sample")

    # --- Nav / platform
    heading = _fallback_1d_aligned(ds, ["heading", "Heading"], pt)
    pitch = _fallback_1d_aligned(ds, ["pitch", "Pitch"], pt)
    roll = _fallback_1d_aligned(ds, ["roll", "Roll"], pt)
    heave = _fallback_1d_aligned(ds, ["heave", "Heave", "vertical_offset"], pt)
    dist = _fallback_1d_aligned(ds, ["dist", "Dist", "distance"], pt)
    n_ping = Sv.sizes["ping_time"]
    zeros = np.zeros(n_ping, dtype=float)
    heading = heading if heading is not None else zeros
    pitch = pitch if pitch is not None else zeros
    roll = roll if roll is not None else zeros
    heave = heave if heave is not None else zeros
    dist = dist if dist is not None else zeros

    # --- Per-channel absorption (optional)
    alpha = 0.0
    if "sound_absorption" in ds:
        sa = ds["sound_absorption"]
        try:
            alpha = (
                float(sa.sel(channel=channel).values.item())
                if "channel" in sa.dims
                else float(sa.values.item())
            )
        except Exception:
            alpha = 0.0

    # --- Transducer depth (fallback 0)
    TD = 0.0
    if "transducer_depth" in ds:
        try:
            td = ds["transducer_depth"].sel(channel=channel)
            TD = float(np.asarray(td).item()) if td.ndim == 0 else float(td.values)
        except Exception:
            TD = 0.0

    # --- Range from transducer face
    R = Depth - (TD + heave.reshape(-1, 1))
    R = R.clip(min=0.0)

    # --- Vertical step & timing
    dstep_da = Depth.isel(ping_time=0).diff("range_sample").median(skipna=True)
    dstep = (
        float(dstep_da.values) if np.isfinite(getattr(dstep_da, "values", np.nan)).all() else np.nan
    )
    dt = (2.0 * dstep) / c if dstep > 0 else 1e-4

    T = float(p.get("pulse_length", 1e-3))
    if "Np" in p and p["Np"] is not None and int(p["Np"]) > 2:
        Np = int(p["Np"])
    else:
        Np = max(3, int(round(T / max(dt, 1e-6))))
    print(Np)
    # --- TS constants
    sv2ts = (
        10.0 * np.log10(c * T / 2.0)
        + float(p["psi_two_way"])
        + 2.0 * float(p["Sa_correction"])
        + 2.0 * float(p["Sa_EK80_nominal"])
    )

    # --- Index domains & bottom cropping
    nb_pings_tot = Sv.sizes["ping_time"]
    nb_samples_tot = Sv.sizes["range_sample"]
    idx_pings_tot = np.arange(nb_pings_tot, dtype=int)
    idx_r_tot = np.arange(nb_samples_tot, dtype=int)

    bb = None
    if bottom_da is not None:
        bb = bottom_da
        if "channel" in bb.dims:
            bb = bb.sel(channel=channel)
        bb = bb.sel(ping_time=Sv["ping_time"])

        D = Depth.values
        b = bb.values
        idx_bot = np.empty(nb_pings_tot, dtype=int)
        for ip in range(nb_pings_tot):
            under = np.nonzero(np.isfinite(D[ip]) & (D[ip] >= b[ip]))[0]
            idx_bot[ip] = under[0] if under.size else (nb_samples_tot - 1)
        max_idx = int(np.nanmax(idx_bot))
        idx_r_tot = np.arange(max_idx + 1, dtype=int)

    # --- Blocking
    cells_per_ping = max(1, idx_r_tot.size)
    block_size = int(min(math.ceil(float(p["block_len"]) / cells_per_ping), idx_pings_tot.size))
    num_ite = int(math.ceil(idx_pings_tot.size / max(block_size, 1)))

    # --- TVG start & helper vectors
    ind_range0 = int(p["tvg_start_sample"])
    ts_range = np.maximum(dstep, (np.arange(nb_samples_tot) - ind_range0) * dstep)  # metres
    NechP = max(3, int(round(T / max(dt, 1e-6))))  # samples per pulse

    # --- Output struct (reuse your function if present)
    try:
        from .esp3 import init_st_struct  # if your helper already lives there
    except Exception:

        def init_st_struct():
            return {
                "TS_comp": [],
                "TS_uncomp": [],
                "Target_range": [],
                "Target_range_disp": [],
                "Target_range_min": [],
                "Target_range_max": [],
                "idx_r": [],
                "StandDev_Angles_Minor_Axis": [],
                "StandDev_Angles_Major_Axis": [],
                "Angle_minor_axis": [],
                "Angle_major_axis": [],
                "Ping_number": [],
                "Time": [],
                "nb_valid_targets": 0,
                "idx_target_lin": [],
                "pulse_env_before_lin": [],
                "pulse_env_after_lin": [],
                "PulseLength_Normalized_PLDL": [],
                "Transmitted_pulse_length": [],
                "Heave": [],
                "Roll": [],
                "Pitch": [],
                "Heading": [],
                "Dist": [],
            }

    out = init_st_struct()
    times = Sv["ping_time"].values

    # --- Parameters for selection
    MaxAngleOW = float(p["MaxAngleOneWayCompression"])
    MinEchoLen = float(p["MinEchoLength"])
    MaxEchoLen = float(p["MaxEchoLength"])
    MinEchoSpace = float(p["MinEchoSpace"])
    MinEchoDepthM = float(p["MinEchoDepthM"])
    MaxEchoDepthM = float(p["MaxEchoDepthM"])
    dec_tir = 8  # ignore first samples near TX

    # Beam pattern params
    bw_al = float(p["beamwidth_along_3dB_rad"])
    bw_at = float(p["beamwidth_athwart_3dB_rad"])
    dec_al = float(p["steer_along_rad"])
    dec_at = float(p["steer_athwart_rad"])

    for ui in range(num_ite):
        start = ui * block_size
        stop = min((ui + 1) * block_size, idx_pings_tot.size)
        idx_pings = idx_pings_tot[start:stop]

        idx_r = idx_r_tot.copy()
        if bb is not None and idx_pings.size > 0:
            D_block = Depth.isel(ping_time=idx_pings).values
            b_block = bb.isel(ping_time=idx_pings).values
            idx_bot = np.empty(idx_pings.size, dtype=int)
            for k in range(idx_pings.size):
                under = np.nonzero(np.isfinite(D_block[k]) & (D_block[k] >= b_block[k]))[0]
                idx_bot[k] = under[0] if under.size else (nb_samples_tot - 1)
            max_idx = int(np.nanmax(idx_bot))
            idx_r = idx_r[idx_r <= max_idx]

        mask = np.zeros((idx_r.size, idx_pings.size), dtype=bool)

        # Subset arrays -> (samples, pings)
        Sv_blk = (
            Sv.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values
        )
        R_blk = (
            R.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values
        )
        DEPblk = (
            Depth.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values
        )
        tsr_blk = ts_range[idx_r].reshape(-1, 1)

        along_blk = None
        athwt_blk = None
        if along is not None:
            along_blk = (
                along.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )
        if athwt is not None:
            athwt_blk = (
                athwt.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )

        # Under-bottom mask
        if bb is not None and idx_pings.size > 0:
            bcol = bb.isel(ping_time=idx_pings).values.reshape(1, -1)
            mask |= DEPblk >= bcol

        # Sv->TSu and Plike
        # tsu = Sv + 20log10(r) + sv2ts ; Plike = tsu - 40log10(r) - 2αr
        r_eff = np.maximum(tsr_blk, 1e-6)
        tsu = Sv_blk + 20.0 * np.log10(r_eff) + sv2ts
        Plike = tsu - 40.0 * np.log10(r_eff) - 2.0 * alpha * tsr_blk

        # Beam comp (one-way approx); if no angles, comp=0
        if along_blk is not None and athwt_blk is not None:
            x = 2.0 * ((along_blk - dec_al) / bw_al)
            y = 2.0 * ((athwt_blk - dec_at) / bw_at)
            one_way_comp = 6.0206 * (x**2 + y**2 - 0.18 * (x**2) * (y**2))
        else:
            one_way_comp = np.zeros_like(tsu)

        ts = tsu + one_way_comp

        # Apply mask
        ts[mask] = np.nan
        tsu[mask] = np.nan
        Plike[mask] = np.nan

        # Per-ping detection
        for j in range(Plike.shape[1]):
            zP = Plike[:, j]
            zTS = ts[:, j]
            zTSU = tsu[:, j]
            rcol = R_blk[:, j]
            if not np.isfinite(zP).any():
                continue

            # local maxima of Plike, skipping first samples
            valid = np.isfinite(zP)
            valid[:dec_tir] = False
            loc = np.where(valid[1:-1] & (zP[1:-1] > zP[:-2]) & (zP[1:-1] >= zP[2:]))[0] + 1

            keep = []
            for k in loc:
                if not np.isfinite(zTS[k]) or zTS[k] <= TS_thr:
                    continue
                # angle comp guard (Matecho uses 2x one-way limit on ts-tsu)
                if (zTS[k] - zTSU[k]) > 2.0 * MaxAngleOW:
                    continue

                # 6-dB width around Plike peak
                base = zP[k]
                i0 = k
                while i0 > 0 and np.isfinite(zP[i0 - 1]) and (zP[i0 - 1] >= base - 6.0):
                    i0 -= 1
                i1 = k
                nP = zP.size
                while i1 + 1 < nP and np.isfinite(zP[i1 + 1]) and (zP[i1 + 1] >= base - 6.0):
                    i1 += 1
                plen = i1 - i0 + 1

                # echo length (in samples) bounds
                if plen < round(NechP * MinEchoLen) or plen > round(NechP * MaxEchoLen):
                    continue

                # min spacing between kept echoes
                if keep and min(abs(k - kk) for kk in keep) < int(round(MinEchoSpace * NechP)):
                    continue

                # depth band test (surface-referenced)
                depth_here = float(rcol[k] + TD + heave[idx_pings[j]])
                if not (MinEchoDepthM <= depth_here <= MaxEchoDepthM):
                    continue

                keep.append(k)

                # Save detection
                r_seg = R_blk[i0 : i1 + 1, j]
                r_seg = r_seg[np.isfinite(r_seg)]
                if r_seg.size == 0:
                    continue
                r_min = float(r_seg.min())
                r_max = float(r_seg.max())
                r_peak = float(rcol[k])

                out["TS_comp"].append(float(zTS[k]))
                out["TS_uncomp"].append(float(zTSU[k]))
                out["Target_range"].append(r_peak)
                out["Target_range_disp"].append(r_peak + c * T / 4.0)
                out["Target_range_min"].append(r_min)
                out["Target_range_max"].append(r_max)

                out["idx_r"].append(int(idx_r[k]))
                out["Ping_number"].append(int(idx_pings[j]))
                out["Time"].append(times[idx_pings[j]])
                out["idx_target_lin"].append(int(idx_pings[j] * nb_samples_tot + idx_r[k]))

                out["pulse_env_before_lin"].append(int(k - i0))
                out["pulse_env_after_lin"].append(int(i1 - k))
                out["PulseLength_Normalized_PLDL"].append(plen / float(NechP))
                out["Transmitted_pulse_length"].append(int(plen))

                out["StandDev_Angles_Minor_Axis"].append(np.nan)
                out["StandDev_Angles_Major_Axis"].append(np.nan)
                out["Angle_minor_axis"].append(np.nan)
                out["Angle_major_axis"].append(np.nan)

                out["Heave"].append(float(heave[idx_pings[j]]))
                out["Roll"].append(float(roll[idx_pings[j]]))
                out["Pitch"].append(float(pitch[idx_pings[j]]))
                out["Heading"].append(float(heading[idx_pings[j]]))
                out["Dist"].append(float(dist[idx_pings[j]]))

    out["nb_valid_targets"] = len(out["TS_comp"])
    return out
