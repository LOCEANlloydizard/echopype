# ---- ESP3 prep: parse/validate, select data, derive indices, blocks, nav ----
import math

import numpy as np
import xarray as xr

DEFAULTS = {
    "SoundSpeed": 1500.0,
    "TS_threshold": -50.0,  # dB
    "PLDL": 6.0,  # dB below peak
    "MinNormPL": 0.7,
    "MaxNormPL": 1.5,
    "DataType": "CW",  # CW only for now
    "block_len": 1e7 / 3,  # ~3.33M cells per block (range*ping)
    # optional:
    # "reg_obj": None,       # (not implemented here)
    # "MaxBeamComp": 4.0,
    # "MaxStdMinAxisAngle": 0.6,
    # "MaxStdMajAxisAngle": 0.6,
}


def init_st_struct():
    """Return a dict mirroring the MATLAB st_struct with empty lists."""
    return {
        "TS_comp": [],
        "TS_uncomp": [],
        "Target_range": [],
        "Target_range_disp": [],
        "Target_range_min": [],
        "Target_range_max": [],
        "idx_r": [],
        "StandDev_Angles_Minor_Axis": [],  # along
        "StandDev_Angles_Major_Axis": [],  # athwart
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


def _validate_params(p: dict):
    if p.get("DataType", "CW") != "CW":
        raise ValueError("Only CW supported at this stage.")
    if not (-120 <= float(p["TS_threshold"]) <= -20):
        raise ValueError("TS_threshold must be in [-120, -20] dB.")
    if not (1 <= float(p["PLDL"]) <= 30):
        raise ValueError("PLDL must be in [1, 30] dB.")
    if not (0.0 <= float(p["MinNormPL"]) <= 10.0):
        raise ValueError("MinNormPL must be in [0, 10].")
    if not (0.0 <= float(p["MaxNormPL"]) <= 10.0):
        raise ValueError("MaxNormPL must be in [0, 10].")
    if float(p["block_len"]) <= 0:
        raise ValueError("block_len must be > 0.")


def detect_esp3(ds: xr.Dataset, params: dict):
    """
    Single-target detection (CW) laid out to mirror the MATLAB structure.
    NOTE: FM branch is stubbed with a TODO.
    """

    # ======================================================================
    # 0) Parse arguments (MATLAB inputParser) + defaults/validation
    # ======================================================================
    if params is None:
        params = {}
    channel = params.get("channel")
    if channel is None:
        raise ValueError("params['channel'] is required.")
    bottom_da = params.get("bottom_da", None)

    # Merge defaults but keep routing keys separate
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k not in ("channel", "bottom_da")}}
    p["DataType"] = p.get("DataType", "CW")  # force CW for now
    _validate_params(p)

    # ======================================================================
    # 1) Select data (trans_obj.* -> ds[...] in our world)
    #    - Get Sv & depth for the chosen channel
    #    - Optionally grab angles (split-beam)
    # ======================================================================

    ######## depth or range sample??????????????

    Sv = ds["Sv"].sel(channel=channel).transpose("ping_time", "range_sample")
    Depth = ds["depth"].sel(channel=channel).transpose("ping_time", "range_sample")

    # ======================================================================

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

    # ======================================================================
    # 2) Build index domains & optional bottom cropping (global idx_r_tot)
    # ======================================================================
    nb_pings_tot = Sv.sizes["ping_time"]
    nb_samples_tot = Sv.sizes["range_sample"]
    idx_pings_tot = np.arange(nb_pings_tot, dtype=int)
    idx_r_tot = np.arange(nb_samples_tot, dtype=int)

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

        # keep rows from 0..max_idx (inclusive)
        max_idx = int(np.nanmax(idx_bot))
        idx_r_tot = np.arange(max_idx + 1, dtype=int)

    # ======================================================================
    # 3) Region/bad-data mask (placeholder)  + block sizing
    # ======================================================================
    # In MATLAB:
    #   idx_bad_data, mask_inter_tot = reg_obj.get_mask_from_intersection(...)
    # Here: we don't have region objects yet, so zeros.
    mask_inter_tot = np.zeros((idx_r_tot.size, idx_pings_tot.size), dtype=bool)

    # Block sizing:
    # block_size = min( ceil(block_len / numel(idx_r_tot)), numel(idx_pings_tot) )
    cells_per_ping = max(1, idx_r_tot.size)
    block_size = int(min(math.ceil(float(p["block_len"]) / cells_per_ping), idx_pings_tot.size))
    num_ite = int(math.ceil(idx_pings_tot.size / block_size)) if block_size > 0 else 0

    # ======================================================================
    # 4) Nav/attitude (use dataset vars if present; else zeros)
    # ======================================================================
    def _pull_nav(ds, name_candidates, ping_time_ref):
        """Return a 1D float array aligned to ping_time_ref (or None)."""
        for nm in name_candidates:
            if nm in ds:
                da = ds[nm]
                # Find the time-like dimension and align to ping_time
                for tdim in ("ping_time", "time", "time1", "time2"):
                    if tdim in da.dims:
                        if tdim != "ping_time":
                            da = da.rename({tdim: "ping_time"})
                        # nearest with tolerance; drop unmatched as NaN (will be filled later)
                        da = da.reindex(
                            ping_time=ping_time_ref,
                            method="nearest",
                            tolerance=np.timedelta64(500, "ms"),
                        )
                        return np.asarray(da.values, dtype=float)
                # If no explicit time dim but same length as pings, just coerce
                if da.ndim == 1 and da.size == ping_time_ref.size:
                    return np.asarray(da.values, dtype=float)
        return None

    nb_pings_tot = (
        ds["Sv"].sel(channel=channel).sizes["ping_time"]
    )  # already defined later; safe here
    pt_ref = Sv["ping_time"]

    heading_arr = _pull_nav(ds, ["heading", "Heading"], pt_ref)
    pitch_arr = _pull_nav(ds, ["pitch", "Pitch"], pt_ref)
    roll_arr = _pull_nav(ds, ["roll", "Roll"], pt_ref)
    heave_arr = _pull_nav(
        ds, ["heave", "Heave", "vertical_offset"], pt_ref
    )  # vertical_offset fallback
    dist_arr = _pull_nav(ds, ["dist", "Dist", "distance"], pt_ref)

    def _fallback(a, n):
        return a if a is not None else np.zeros(n, dtype=float)

    heading = _fallback(heading_arr, nb_pings_tot)
    pitch = _fallback(pitch_arr, nb_pings_tot)
    roll = _fallback(roll_arr, nb_pings_tot)
    heave = _fallback(heave_arr, nb_pings_tot)
    dist = _fallback(dist_arr, nb_pings_tot)

    # ======================================================================
    # 5) Time & static vectors
    # ======================================================================
    times = Sv["ping_time"].values
    range_vec = Depth.isel(ping_time=0).values  # informative only

    # ======================================================================
    # 6) Prepare constants (alpha, c, pulse length Np, T)
    # ======================================================================
    # sound absorption alpha (per channel if provided)
    alpha = 0.0
    if "sound_absorption" in ds:
        print("found sound_absorption")
        sa = ds["sound_absorption"]
        try:
            alpha = (
                float(sa.sel(channel=channel).values.item())
                if "channel" in sa.dims
                else float(sa.values.item())
            )
        except Exception:
            alpha = 0.0
    else:
        print("did not find sound_sbasoprtion")

    c = float(p["SoundSpeed"])

    # Determine Np (samples) and T (seconds)
    if "Np" in p and p["Np"] is not None and int(p["Np"]) > 2:
        Np = int(p["Np"])
        T = float(p.get("pulse_length", 1e-3))
    else:
        dstep_da = Depth.diff("range_sample").median(skipna=True)
        dstep = (
            float(dstep_da.values.item())
            if getattr(dstep_da, "size", 1) == 1
            else float(dstep_da.values)
        )
        dt = (2.0 * dstep) / c if dstep > 0 else 1e-4
        T = float(p.get("pulse_length", 1e-3))
        Np = max(3, int(round(T / dt)))

    TS_thr = float(p["TS_threshold"])
    PLDL = float(p["PLDL"])
    min_len = max(1, int(Np * float(p["MinNormPL"])))
    max_len = max(1, int(math.ceil(Np * float(p["MaxNormPL"]))))

    # ======================================================================
    # 7) Init output struct (MATLAB single_targets_tot)
    # ======================================================================
    out = init_st_struct()

    # Prepare bottom per-ping (channel sliced) once
    bb = None
    if bottom_da is not None:
        bb = bottom_da
        if "channel" in bb.dims:
            bb = bb.sel(channel=channel)
        bb = bb.sel(ping_time=Sv["ping_time"])

    # ======================================================================
    # 8) Block loop  (for ui = 1 : num_ite)
    # ======================================================================

    for ui in range(num_ite):
        # 8.1) Select ping block
        start = ui * block_size
        stop = min((ui + 1) * block_size, idx_pings_tot.size)
        idx_pings = idx_pings_tot[start:stop]

        # 8.2) Per-block row set (crop to per-block bottom inclusive)
        idx_r = idx_r_tot.copy()
        if bb is not None and idx_pings.size > 0:
            D_block = Depth.isel(ping_time=idx_pings).values
            b_block = bb.isel(ping_time=idx_pings).values
            idx_bot = np.empty(idx_pings.size, dtype=int)
            for k in range(idx_pings.size):
                under = np.nonzero(np.isfinite(D_block[k]) & (D_block[k] >= b_block[k]))[0]
                idx_bot[k] = under[0] if under.size else (nb_samples_tot - 1)
            max_idx = int(np.nanmax(idx_bot))
            idx_r = idx_r[idx_r <= max_idx]  # inclusive crop

        # 8.3) Build mask (regions + under-bottom later)
        mask = np.zeros((idx_r.size, idx_pings.size), dtype=bool)
        if mask_inter_tot.size:
            r_pos = np.searchsorted(idx_r_tot, idx_r)
            p_pos = np.searchsorted(idx_pings_tot, idx_pings)
            mask |= mask_inter_tot[np.ix_(r_pos, p_pos)]

        # 8.4) Extract sub-matrices (TS/Depth) as [samples, pings]
        TS = (
            Sv.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values.copy()
        )
        DEP = (
            Depth.isel(ping_time=idx_pings, range_sample=idx_r)
            .transpose("range_sample", "ping_time")
            .values
        )

        # Pre-align angle matrices to current block (same shape as TS, DEP)
        print("okkkkkkkkkkk")
        along_block = None
        athwt_block = None
        if along is not None:
            along_block = (
                along.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )
        if athwt is not None:
            athwt_block = (
                athwt.isel(ping_time=idx_pings, range_sample=idx_r)
                .transpose("range_sample", "ping_time")
                .data
            )

        print("ok")

        # 8.5) Under-bottom mask
        if bb is not None and idx_pings.size > 0:
            bcol = bb.isel(ping_time=idx_pings).values.reshape(1, -1)
            mask |= DEP >= bcol

        # 8.6) Apply mask => TS = -999 (MATLAB convention)
        TS[mask] = -999.0
        if not np.any(TS > -999.0):
            continue

        # 8.7) Remove trailing all-masked rows
        valid_rows = np.where((TS > -999.0).any(axis=1))[0]
        last_row = valid_rows.max()
        row_sel = slice(0, last_row + 1)
        TS = TS[row_sel, :]
        DEP = DEP[row_sel, :]
        idx_r = idx_r[row_sel]

        nb_samples, nb_pings = TS.shape

        # 8.8) Compute TVG and (optionally) Power like MATLAB
        # (we run a TS-based detector below; Power left here for parity)
        r_eff = DEP - c * T / 4.0
        TVG = np.where(r_eff > 0, 40.0 * np.log10(r_eff) + 2.0 * alpha * r_eff, np.nan)
        # Power = TS - TVG

        # ==================================================================
        # 9) Peak picking & PLDL expansion  (CW branch)
        #    (FM branch placeholder)
        # ==================================================================
        if p["DataType"] == "CW":
            # Minimal TS-based detector:
            for jp in range(nb_pings):
                z = TS[:, jp]  # TS (dB)
                d = DEP[:, jp]  # depth (m)
                if z.size < 3:
                    continue

                # local maxima (strict > prev & >= next), skipping masked rows
                valid = z > -999.0
                cand = np.where(valid[1:-1] & (z[1:-1] > z[:-2]) & (z[1:-1] >= z[2:]))[0] + 1

                # min separation = Np
                keep, last = [], -(10**9)
                for k in cand:
                    if k - last >= Np:
                        keep.append(k)
                        last = k

                for k in keep:
                    pk = float(z[k])
                    if not np.isfinite(pk) or pk <= TS_thr:
                        continue
                    thr = pk - PLDL

                    # expand left/right while above threshold
                    left = 0
                    i = k - 1
                    while i >= 0 and left < max_len and np.isfinite(z[i]) and z[i] >= thr:
                        left += 1
                        i -= 1
                    right = 0
                    i = k + 1
                    while i < z.size and right < max_len and np.isfinite(z[i]) and z[i] >= thr:
                        right += 1
                        i += 1

                    plen = 1 + left + right
                    if plen < min_len or plen > max_len:
                        continue

                    i0, i1 = k - left, k + right
                    seg = d[i0 : i1 + 1]
                    seg = seg[np.isfinite(seg)]
                    if seg.size == 0:
                        continue

                    r_min = float(seg.min())
                    r_max = float(seg.max())
                    r_peak = float(d[k]) if np.isfinite(d[k]) else 0.5 * (r_min + r_max)

                    # TVG at peak (uncomp -> comp placeholder)
                    r_eff_peak = r_peak - c * T / 4.0
                    if r_eff_peak <= 0:
                        continue
                    TVG_peak = 40.0 * np.log10(r_eff_peak) + 2.0 * alpha * r_eff_peak
                    TS_uncomp = pk + TVG_peak
                    TS_comp = TS_uncomp  # no beam comp yet (simradBeamCompensation TODO)

                    if TS_comp <= TS_thr:
                        continue

                    # -------------------------------------------------------
                    # ADD angle + nav stats here (per detection)
                    # -------------------------------------------------------

                    # print("mmmmhmmm m'okay")
                    # ping_glob = int(idx_pings[jp])  # for nav arrays

                    # # Angle at the peak sample only (no window stats)
                    # if along_block is not None:
                    #     ang_along_peak = float(along_block[k, jp])
                    #     if not np.isfinite(ang_along_peak):
                    #         ang_along_peak = np.nan
                    # else:
                    #     ang_along_peak = np.nan

                    # if athwt_block is not None:
                    #     ang_athwt_peak = float(athwt_block[k, jp])
                    #     if not np.isfinite(ang_athwt_peak):
                    #         ang_athwt_peak = np.nan
                    # else:
                    #     ang_athwt_peak = np.nan

                    # print("m'okay")

                    # --- Save detection (global indexing like MATLAB) ---
                    out["TS_comp"].append(TS_comp)
                    out["TS_uncomp"].append(TS_uncomp)
                    out["Target_range"].append(r_peak)
                    out["Target_range_disp"].append(r_peak + c * T / 4.0)
                    out["Target_range_min"].append(r_min)
                    out["Target_range_max"].append(r_max)

                    out["idx_r"].append(int(idx_r[k]))  # global range_sample index
                    out["Ping_number"].append(int(idx_pings[jp]))  # global ping index
                    out["Time"].append(times[idx_pings[jp]])  # ping_time stamp
                    out["idx_target_lin"].append(int(idx_pings[jp] * nb_samples_tot + idx_r[k]))

                    out["pulse_env_before_lin"].append(int(left))
                    out["pulse_env_after_lin"].append(int(right))
                    out["PulseLength_Normalized_PLDL"].append(plen / float(Np))
                    out["Transmitted_pulse_length"].append(int(plen))

                    # angles
                    out["StandDev_Angles_Minor_Axis"].append(np.nan)
                    out["StandDev_Angles_Major_Axis"].append(np.nan)
                    # out["Angle_minor_axis"].append(ang_along_peak)   # alongship @ peak => change to mean ?
                    # out["Angle_major_axis"].append(ang_athwt_peak)   # athwartship @ peak => change to mean ?

                    # # nav (cheap 1D lookups)
                    # out["Heave"].append(float(heave[ping_glob]))
                    # out["Roll"].append(float(roll[ping_glob]))
                    # out["Pitch"].append(float(pitch[ping_glob]))
                    # out["Heading"].append(float(heading[ping_glob]))
                    # out["Dist"].append(float(dist[ping_glob]))

        else:
            # --------------------------- FM TODO ---------------------------
            # In MATLAB they smooth peak_mat over ~Np/2 window, then
            # islocalmax with MinSeparation = Np/2.
            # WE'll replicate the same pattern here once FM is supported.
            # ---------------------------------------------------------------
            pass

    # ======================================================================
    # 10) Finalize / return (MATLAB sums nb_valid_targets)
    # ======================================================================
    out["nb_valid_targets"] = len(out["TS_comp"])
    return out
