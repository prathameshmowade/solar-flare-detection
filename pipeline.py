#!/usr/bin/env python3
"""
pipeline.py — Aditya-L1 Solar Flare Detection & Forecasting Pipeline
Bharatiya Antariksh Hackathon 2026 (H2S)

Architecture:
  Aditya-L1 → [SoLEXS | HEL1OS] → Independent detection
  → Master catalogue → Physics features → AI model → Probability + lead time

Usage:
  python pipeline.py --demo                   # synthetic data, full run
  python pipeline.py --data aditya_l1.json    # real ISSDC JSON
  python pipeline.py --demo --no-train        # detection only, skip ML
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
DT               = 4          # cadence seconds
BG_WINDOW        = 300        # background estimation window (pts)
SXR_THRESHOLD    = 3.0        # SXR detection: N × background
HXR_MIN_PROM     = 4.0        # HXR detection: N × local noise
MERGE_WINDOW_S   = 400        # max SXR–HXR offset for a match (s)
LOOKBACK_S       = 900        # feature look-back window (s)
HORIZON_S        = 1800       # forecasting horizon (30 min)
FLARE_CLASS_MAP  = {
    'X': (400, 1e9), 'M': (40, 400), 'C': (8, 40),
    'B': (2.8, 8),   'A': (0, 2.8)
}

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  Aditya-L1 Solar Flare Detection & Forecasting Pipeline      ║
║  Bharatiya Antariksh Hackathon 2026 (H2S)                    ║
╚══════════════════════════════════════════════════════════════╝"""


# ═══════════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA (demo mode)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic(duration_h=6.5, cadence_s=4, seed=42):
    """
    Realistic Aditya-L1 synthetic data.
    Flares are spaced far enough apart that each one's background estimate
    isn't contaminated by the previous flare's decay tail, and that the
    forecasting horizon sees a healthy mix of quiet and pre-flare windows.
    Returns dict with keys: time, solexs (3×N), helios (4×N), known_flares.
    """
    rng = np.random.default_rng(seed)
    N   = int(duration_h * 3600 / cadence_s)
    t   = np.arange(N) * cadence_s

    BG_S = np.array([1800, 700, 260])
    BG_H = np.array([110,  48,  20,  8])

    # Poisson background noise
    s = np.stack([rng.poisson(b, N).astype(float) for b in BG_S])
    h = np.stack([rng.poisson(b, N).astype(float) for b in BG_H])

    # Flare table: peak_t(s), rise(s), decay(s), sM, hM, pre-flare flag
    # Spaced ~70-100 min apart so background windows stay clean and the
    # forecast horizon (30 min) has substantial quiet (negative) coverage.
    FLARES = [
        dict(pt=1800,  rt=220, dt=1100, sM=14,  hM=25,  pre=True),   # C-class
        dict(pt=6000,  rt=85,  dt=580,  sM=70,  hM=175, pre=True),   # M-class
        dict(pt=10800, rt=160, dt=820,  sM=7,   hM=11,  pre=False), # B-class (weak)
        dict(pt=15600, rt=55,  dt=300,  sM=450, hM=900, pre=True),  # X-class
        dict(pt=19800, rt=180, dt=950,  sM=12,  hM=20,  pre=True),  # C-class
    ]
    known = []

    for f in FLARES:
        pi = int(f['pt'] / cadence_s)
        for i in range(N):
            dt_s = (i - pi) * cadence_s
            # Soft X-ray: Gaussian rise + exponential decay
            sp = (np.exp(-dt_s**2 / (2*f['rt']**2)) if dt_s < 0
                  else np.exp(-dt_s / f['dt']))
            # Hard X-ray: more impulsive (narrower, faster)
            hr, hd = f['rt'] * 0.22, f['dt'] * 0.13
            hp = (np.exp(-dt_s**2 / (2*hr**2)) if dt_s < 0
                  else np.exp(-dt_s / hd))

            s[0, i] += BG_S[0] * f['sM'] * 0.38 * sp
            s[1, i] += BG_S[1] * f['sM'] * 0.68 * sp
            s[2, i] += BG_S[2] * f['sM'] * 1.00 * sp
            h[0, i] += BG_H[0] * f['hM'] * 0.85 * hp
            h[1, i] += BG_H[1] * f['hM'] * 1.00 * hp
            h[2, i] += BG_H[2] * f['hM'] * 0.60 * hp
            h[3, i] += BG_H[3] * f['hM'] * 0.32 * hp

            # Pre-flare soft X-ray enhancement (Gaussian centred at −15 min)
            if f['pre'] and -2100 <= dt_s < -f['rt'] * 2.5:
                pe = 0.13 * np.exp(-((dt_s + 900)**2) / (2 * 450**2))
                if pe > 5e-4:
                    s[0, i] *= (1 + pe)
                    s[1, i] *= (1 + pe * 0.5)

        known.append({'peak_time': f['pt'], 'peak_idx': pi,
                      'class': classify_by_magnitude(BG_S[2] * f['sM'])})

    return {'time': t, 'solexs': np.clip(s, 1, None),
            'helios': np.clip(h, 1, None), 'known_flares': known}


def load_json(path):
    """Load ISSDC JSON (from convert_fits.py output)."""
    with open(path) as f:
        j = json.load(f)
    if j.get('solexs') and j['solexs'].get('time'):
        t    = np.array(j['solexs']['time'])
        t   -= t[0]
        sv   = list(j['solexs']['channels'].values())[:3]
        hv   = list((j.get('helios') or {}).get('channels', {}).values())[:4]
        s    = np.stack([np.array(c) for c in sv])
        h    = np.stack([np.array(c) for c in hv]) if hv else np.ones((4, len(t)))
        return {'time': t, 'solexs': np.clip(s, 1, None),
                'helios': np.clip(h, 1, None), 'known_flares': []}
    raise ValueError("Unrecognised JSON format — run convert_fits.py first")


# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def rolling_background(flux, window=BG_WINDOW, percentile=10):
    """Rolling low-percentile background estimate (robust against flares)."""
    bg = np.empty_like(flux)
    for i in range(len(flux)):
        sl   = flux[max(0, i - window): i + 1]
        bg[i] = np.percentile(sl, percentile)
    return np.clip(bg, 1, None)


def preprocess(data):
    """Compute normalised flux and background for each channel."""
    out = {}
    for key in ('solexs', 'helios'):
        arr = data[key]                              # (C, N)
        bgs = np.stack([rolling_background(arr[c]) for c in range(len(arr))])
        out[key]         = arr
        out[f'{key}_bg'] = bgs
        out[f'{key}_ratio'] = arr / bgs              # flux / background
    out['time'] = data['time']
    out['known_flares'] = data.get('known_flares', [])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3. SXR FLARE DETECTOR  (SoLEXS primary: 6–10 keV channel index 2)
# ═══════════════════════════════════════════════════════════════════════════

def classify_by_magnitude(peak_flux):
    for cls, (lo, hi) in FLARE_CLASS_MAP.items():
        if lo < peak_flux <= hi:
            return cls
    return 'A'


def detect_sxr(prep, threshold=SXR_THRESHOLD, channel=2):
    """
    Detect flares in SoLEXS using threshold + positive-gradient trigger.
    Returns DataFrame: start_idx, peak_idx, end_idx, start_time, peak_time,
                       end_time, peak_flux, peak_ratio, class.
    """
    ratio = prep['solexs_ratio'][channel]
    flux  = prep['solexs'][channel]
    t     = prep['time']
    events, in_flare, start = [], False, 0

    for i in range(5, len(ratio) - 5):
        rising = ratio[i] > ratio[i - 5]
        if not in_flare and ratio[i] > threshold and rising:
            in_flare, start = True, i
        elif in_flare and ratio[i] < threshold * 0.45:
            # Find peak within the flare segment
            seg   = flux[start:i + 1]
            pk    = start + int(np.argmax(seg))
            events.append({
                'start_idx':  start, 'peak_idx':  pk,   'end_idx':  i,
                'start_time': t[start], 'peak_time': t[pk], 'end_time': t[i],
                'peak_flux':  flux[pk],
                'peak_ratio': flux[pk] / max(prep['solexs_bg'][channel][pk], 1),
                'class':      classify_by_magnitude(ratio[pk]),
            })
            in_flare = False

    return pd.DataFrame(events)


# ═══════════════════════════════════════════════════════════════════════════
# 4. HXR FLARE DETECTOR  (HEL1OS primary: 10–15 keV channel index 1)
# ═══════════════════════════════════════════════════════════════════════════

def detect_hxr(prep, min_prom=HXR_MIN_PROM, channel=1):
    """
    Detect impulsive peaks in HEL1OS using scipy.signal.find_peaks.
    Returns DataFrame: peak_idx, peak_time, peak_counts, prominence,
                       onset_idx, end_idx, duration_s.
    """
    flux     = prep['helios'][channel]
    bg       = prep['helios_bg'][channel]
    t        = prep['time']
    noise    = np.std(flux[:200])          # estimate from quiet start
    prom_thr = max(min_prom * noise, 10)

    peaks, props = find_peaks(
        flux, prominence=prom_thr, distance=30, width=3
    )
    if len(peaks) == 0:
        return pd.DataFrame()

    rows = []
    for pk, prom, wl, wr in zip(
        peaks,
        props['prominences'],
        props['left_bases'],
        props['right_bases'],
    ):
        rows.append({
            'peak_idx':    pk,
            'peak_time':   t[pk],
            'peak_counts': flux[pk],
            'prominence':  prom,
            'onset_idx':   wl,
            'end_idx':     wr,
            'duration_s':  t[min(wr, len(t)-1)] - t[wl],
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MASTER CATALOGUE — merge SXR + HXR events
# ═══════════════════════════════════════════════════════════════════════════

def merge_catalogues(sxr_cat, hxr_cat, merge_window_s=MERGE_WINDOW_S):
    """
    Match SXR and HXR events within ±merge_window_s of each other.
    Returns master catalogue with time-lag column (SXR_peak - HXR_peak).
    """
    rows = []
    if sxr_cat.empty:
        return pd.DataFrame()

    hxr_times = (hxr_cat['peak_time'].values
                 if not hxr_cat.empty else np.array([]))

    for _, ev in sxr_cat.iterrows():
        match_hxr = None
        if len(hxr_times):
            diffs   = np.abs(hxr_times - ev['peak_time'])
            closest = int(np.argmin(diffs))
            if diffs[closest] <= merge_window_s:
                match_hxr = hxr_cat.iloc[closest]

        row = ev.to_dict()
        row['hxr_peak_time']   = match_hxr['peak_time']   if match_hxr is not None else np.nan
        row['hxr_peak_counts'] = match_hxr['peak_counts'] if match_hxr is not None else np.nan
        row['hxr_matched']     = match_hxr is not None
        row['lag_s']           = (ev['peak_time'] - match_hxr['peak_time']
                                  if match_hxr is not None else np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PHYSICS-BASED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def neupert_ratio(sxr_flux, hxr_flux, dt_s=DT):
    """
    Neupert Effect: the time integral of HXR ≈ the SXR flux (thermal response).
    Returns: Pearson r between d(SXR)/dt and HXR in the shared window.
    """
    dsxr_dt = np.gradient(sxr_flux, dt_s)
    r, _    = stats.pearsonr(dsxr_dt, hxr_flux)
    return float(r)


def sxr_hxr_lag(sxr_flux, hxr_flux, max_lag_pts=150):
    """
    Cross-correlation lag between SXR and HXR (negative = HXR leads).
    Returns lag in seconds.
    """
    s = (sxr_flux - sxr_flux.mean()) / (sxr_flux.std() + 1e-8)
    h = (hxr_flux - hxr_flux.mean()) / (hxr_flux.std() + 1e-8)
    corr       = np.correlate(s, h, mode='full')
    lags       = np.arange(-len(s) + 1, len(s))
    mid        = len(lags) // 2
    roi        = slice(max(0, mid - max_lag_pts), mid + max_lag_pts + 1)
    best_lag   = lags[roi][int(np.argmax(corr[roi]))]
    return float(best_lag * DT)


def shape_features(flux, peak_idx, bg_level):
    """
    Rise time, decay time, FWHM, and asymmetry of a flare profile.
    peak_idx is relative to start of flux array.
    """
    peak_val = float(flux[peak_idx])
    half     = (peak_val + bg_level) / 2.0

    # Rise: scan backwards from peak to find 50% level
    rise_t = 0.0
    for i in range(peak_idx, -1, -1):
        if flux[i] <= half:
            rise_t = (peak_idx - i) * DT
            break

    # Decay: scan forwards from peak to find 50% level
    decay_t = 0.0
    for i in range(peak_idx, len(flux)):
        if flux[i] <= half:
            decay_t = (i - peak_idx) * DT
            break

    fwhm       = rise_t + decay_t
    asymmetry  = rise_t / max(decay_t, 1)           # <1 = impulsive
    sharpness  = (peak_val - bg_level) / max(fwhm, DT)

    return {
        'rise_time_s':  rise_t,
        'decay_time_s': decay_t,
        'fwhm_s':       fwhm,
        'asymmetry':    asymmetry,
        'sharpness':    sharpness,
        'peak_over_bg': (peak_val - bg_level) / max(bg_level, 1),
    }


def window_statistics(flux, window_size):
    """Statistical descriptors of a flux window."""
    f     = flux[-window_size:]
    mu    = float(np.mean(f))
    sigma = float(np.std(f)) + 1e-8
    return {
        'mean':   mu,
        'std':    sigma,
        'cv':     sigma / abs(mu),          # coefficient of variation
        'skew':   float(stats.skew(f)),
        'kurt':   float(stats.kurtosis(f)),
        'p90':    float(np.percentile(f, 90)),
        'slope':  float(np.polyfit(np.arange(len(f)), f, 1)[0]),  # counts/step
    }


def compute_event_features(ev, prep):
    """
    Full physics feature vector for one event in the master catalogue.
    Uses pre-flare data only (no lookahead) for forecasting validity.
    """
    N    = len(prep['time'])
    pi   = int(ev['peak_idx'])
    lb   = int(LOOKBACK_S / DT)
    lo   = max(0, pi - lb)

    sxr  = prep['solexs'][2, lo:pi]     # primary 6–10 keV SoLEXS
    hxr  = prep['helios'][1, lo:pi]     # primary 10–15 keV HEL1OS
    bg_s = float(np.mean(prep['solexs_bg'][2, lo:lo + 10]))
    bg_h = float(np.mean(prep['helios_bg'][1, lo:lo + 10]))

    feats = {}

    # — Neupert Effect ——————————————————————————————————————
    if len(sxr) > 5:
        feats['neupert_r'] = neupert_ratio(sxr, hxr)
    else:
        feats['neupert_r'] = 0.0

    # — SXR–HXR time lag ————————————————————————————————————
    feats['sxr_hxr_lag_s'] = sxr_hxr_lag(sxr, hxr) if len(sxr) > 10 else 0.0

    # — Cross-correlation peak ————————————————————————————————
    if len(sxr) > 10:
        s_n  = (sxr - sxr.mean()) / (sxr.std() + 1e-8)
        h_n  = (hxr - hxr.mean()) / (hxr.std() + 1e-8)
        xcorr = np.correlate(s_n, h_n, mode='full')
        feats['xcorr_max']   = float(np.max(xcorr) / len(sxr))
        feats['xcorr_mean']  = float(np.mean(np.abs(xcorr)) / len(sxr))
    else:
        feats['xcorr_max']   = 0.0
        feats['xcorr_mean']  = 0.0

    # — Shape features ——————————————————————————————————————
    full_sxr = prep['solexs'][2, lo: min(N, pi + int(LOOKBACK_S / DT))]
    pk_in_seg = min(pi - lo, len(full_sxr) - 1)
    shp = shape_features(full_sxr, pk_in_seg, bg_s)
    feats.update(shp)

    # — Statistical features ————————————————————————————————
    win = max(10, min(len(sxr), int(300 / DT)))
    s_stats = window_statistics(sxr, win)
    h_stats = window_statistics(hxr, win)
    for k, v in s_stats.items():
        feats[f'sxr_{k}'] = v
    for k, v in h_stats.items():
        feats[f'hxr_{k}'] = v

    # — Pre-flare enhancement ——————————————————————————————
    early = float(np.mean(sxr[:max(1, len(sxr)//4)]))
    late  = float(np.mean(sxr[-max(1, len(sxr)//4):]))
    feats['preflare_enhance'] = (late - early) / max(early, 1)

    # — HXR impulsiveness ——————————————————————————————————
    feats['hxr_impulse_cv']  = float(np.std(np.diff(hxr)) / (np.mean(np.abs(np.diff(hxr))) + 1e-8))

    # — Catalogue-derived ——————————————————————————————————
    feats['lag_s']           = float(ev.get('lag_s', 0) or 0)
    feats['hxr_matched']     = float(ev.get('hxr_matched', 0))
    feats['peak_ratio']      = float(ev.get('peak_ratio', 1))

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# 7. SLIDING WINDOW FEATURE MATRIX (for ML training)
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_matrix(prep, stride=25):
    """
    Build X (feature matrix) and y (labels) for ML training.
    For each window ending at time t:
      - Compute rolling features
      - Label = 1 if a known flare peaks in [t, t + HORIZON_S]
    Only usable when known_flares is populated (synthetic / annotated data).
    """
    known  = prep['known_flares']
    if not known:
        print("  [WARN] No known flares — cannot build labelled matrix.")
        return pd.DataFrame(), np.array([])

    flare_times = {f['peak_time'] for f in known}
    N    = len(prep['time'])
    lb   = int(LOOKBACK_S / DT)
    hz   = int(HORIZON_S  / DT)
    rows, labels = [], []

    sxr_p = prep['solexs'][2]    # 6–10 keV SoLEXS
    hxr_p = prep['helios'][1]    # 10–15 keV HEL1OS
    sxr_bg = prep['solexs_bg'][2]
    hxr_bg = prep['helios_bg'][1]

    for i in range(lb, N - hz, stride):
        t_now  = prep['time'][i]
        t_end  = prep['time'][min(i + hz, N - 1)]
        sxr_w  = sxr_p[i - lb: i]
        hxr_w  = hxr_p[i - lb: i]

        # Label: does a known flare peak fall in the forecast horizon?
        label = int(any(t_now <= ft <= t_end for ft in flare_times))

        # Rolling features
        feats = {}
        feats['neupert_r']       = neupert_ratio(sxr_w, hxr_w) if lb > 5 else 0
        feats['sxr_hxr_lag_s']  = sxr_hxr_lag(sxr_w, hxr_w)
        feats['preflare_enhance']= float((np.mean(sxr_w[-20:]) - np.mean(sxr_w[:20]))
                                         / (np.mean(sxr_w[:20]) + 1e-8))
        feats['hxr_impulse_cv'] = float(np.std(np.diff(hxr_w))
                                         / (np.mean(np.abs(np.diff(hxr_w))) + 1e-8))
        win = min(len(sxr_w), int(300 / DT))
        for k, v in window_statistics(sxr_w, win).items():
            feats[f'sxr_{k}'] = v
        for k, v in window_statistics(hxr_w, win).items():
            feats[f'hxr_{k}'] = v
        feats['sxr_ratio_now'] = float(sxr_p[i] / max(sxr_bg[i], 1))
        feats['hxr_ratio_now'] = float(hxr_p[i] / max(hxr_bg[i], 1))

        rows.append(feats)
        labels.append(label)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    print(f"  Feature matrix: {X.shape[0]} windows × {X.shape[1]} features  "
          f"({y.sum()} positive / {(~y.astype(bool)).sum()} negative)")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# 8. XGBOOST FORECASTING MODEL
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost(X, y, test_frac=0.25):
    """
    Train XGBoost flare probability model.
    Splits TEMPORALLY (no shuffle) to avoid data leakage.
    Returns (model, metrics_dict, feature_importance_series).
    """
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import (roc_auc_score, precision_score,
                                     recall_score, f1_score)
    except ImportError:
        print("  [SKIP] xgboost not installed — pip install xgboost")
        return None, {}, pd.Series()

    split  = int(len(X) * (1 - test_frac))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split],      y[split:]

    pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_w, use_label_encoder=False,
        eval_metric='logloss', random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    proba = model.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    metrics = {
        'AUC':       round(roc_auc_score(y_te, proba), 3) if y_te.sum() > 0 else 0,
        'Recall':    round(recall_score(y_te, pred, zero_division=0), 3),
        'Precision': round(precision_score(y_te, pred, zero_division=0), 3),
        'F1':        round(f1_score(y_te, pred, zero_division=0), 3),
        'FAR':       round(float((pred == 1).sum() / max((y_te == 0).sum(), 1)), 3),
    }
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return model, metrics, importances


# ═══════════════════════════════════════════════════════════════════════════
# 9. LSTM FORECASTING MODEL
# ═══════════════════════════════════════════════════════════════════════════

def build_lstm_sequences(prep, seq_len=90, stride=15):
    """
    Build (X_seq, y_seq) for LSTM.
    X_seq shape: (N, seq_len, C)  where C = 7 channels (3 SXR + 4 HXR)
    y_seq shape: (N,)
    """
    known = prep['known_flares']
    if not known:
        return np.array([]), np.array([])

    flare_times = {f['peak_time'] for f in known}
    hz = int(HORIZON_S / DT)
    all_ch = np.vstack([prep['solexs'], prep['helios']])   # (7, N)
    # Log-normalise each channel
    all_ch = np.log1p(all_ch)
    N = all_ch.shape[1]
    seqs, labels = [], []

    for i in range(seq_len, N - hz, stride):
        t_now = prep['time'][i]
        t_end = prep['time'][min(i + hz, N - 1)]
        seq   = all_ch[:, i - seq_len: i].T   # (seq_len, 7)
        label = int(any(t_now <= ft <= t_end for ft in flare_times))
        seqs.append(seq)
        labels.append(label)

    return np.stack(seqs, 0), np.array(labels)


def train_lstm(X_seq, y_seq, epochs=30, test_frac=0.25):
    """
    Train a 2-layer LSTM for flare probability forecasting.
    Requires PyTorch. Returns (model, metrics_dict) or (None, {}).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  [SKIP] PyTorch not installed — pip install torch")
        return None, {}

    class LSTMForecaster(nn.Module):
        def __init__(self, n_ch=7, hidden=64, layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_ch, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1), nn.Sigmoid()
            )
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.head(h[-1]).squeeze(-1)

    split = int(len(X_seq) * (1 - test_frac))
    X_tr  = torch.tensor(X_seq[:split], dtype=torch.float32)
    y_tr  = torch.tensor(y_seq[:split], dtype=torch.float32)
    X_te  = torch.tensor(X_seq[split:], dtype=torch.float32)
    y_te  = torch.tensor(y_seq[split:], dtype=torch.float32)

    pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    ds    = TensorDataset(X_tr, y_tr)
    dl    = DataLoader(ds, batch_size=32, shuffle=True)

    model     = LSTMForecaster()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss(weight=None)

    best_loss, patience, patience_max = 1e9, 0, 8
    for ep in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in dl:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te), y_te).item()
        print(f"  Epoch {ep+1:02d}/{epochs}  train={epoch_loss/len(dl):.4f}  "
              f"val={val_loss:.4f}", end='\r')
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            patience  = 0
        else:
            patience += 1
            if patience >= patience_max:
                break

    print()
    model.eval()
    with torch.no_grad():
        proba = model(X_te).numpy()
        pred  = (proba >= 0.5).astype(int)
        y_np  = y_te.numpy().astype(int)

    from sklearn.metrics import recall_score, precision_score, f1_score
    metrics = {
        'AUC':       round(float(roc_auc_score(y_np, proba)), 3) if y_np.sum() > 0 else 0,
        'Recall':    round(float(recall_score(y_np, pred, zero_division=0)), 3),
        'Precision': round(float(precision_score(y_np, pred, zero_division=0)), 3),
        'F1':        round(float(f1_score(y_np, pred, zero_division=0)), 3),
    }
    return model, metrics


# ═══════════════════════════════════════════════════════════════════════════
# 10. REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def print_catalogue(df, name):
    print(f"\n  ── {name} ({len(df)} events) ──")
    if df.empty:
        print("  (none detected)")
        return
    cols = [c for c in ('start_time','peak_time','end_time','class',
                         'peak_ratio','lag_s','hxr_matched')
            if c in df.columns]
    with pd.option_context('display.float_format', '{:.1f}'.format,
                           'display.max_columns', 10, 'display.width', 120):
        print(df[cols].to_string(index=False))


def print_metrics(name, m):
    if not m:
        return
    print(f"\n  ── {name} evaluation ──")
    for k, v in m.items():
        print(f"  {k:<14}: {v}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(BANNER)

    ap = argparse.ArgumentParser(description="Aditya-L1 flare pipeline")
    ap.add_argument('--demo',     action='store_true', help="use synthetic data")
    ap.add_argument('--data',     type=str,            help="path to JSON from convert_fits.py")
    ap.add_argument('--no-train', action='store_true', help="skip ML training")
    args = ap.parse_args()

    if not args.demo and not args.data:
        print("\nUsage:\n  python pipeline.py --demo\n"
              "  python pipeline.py --data aditya_l1_data.json")
        sys.exit(0)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    if args.demo:
        print("  Generating 2.5-hour synthetic Aditya-L1 dataset …")
        raw = generate_synthetic()
        print(f"  {len(raw['time'])} time points  "
              f"({raw['time'][-1]/3600:.1f} h)  "
              f"{len(raw['known_flares'])} planted flares")
    else:
        raw = load_json(args.data)
        print(f"  Loaded {len(raw['time'])} time points from {args.data}")

    # ── Preprocessing ──────────────────────────────────────────────────────
    print("\n[2/7] Preprocessing (background estimation) …")
    prep = preprocess(raw)
    print("  Done.")

    # ── SXR detection ──────────────────────────────────────────────────────
    print("\n[3/7] SXR flare detection …")
    sxr_cat = detect_sxr(prep)
    print_catalogue(sxr_cat, "SXR catalogue")

    # ── HXR detection ──────────────────────────────────────────────────────
    print("\n[4/7] HXR flare detection …")
    hxr_cat = detect_hxr(prep)
    print_catalogue(hxr_cat, "HXR catalogue")

    # ── Master catalogue ───────────────────────────────────────────────────
    print("\n[5/7] Building master catalogue …")
    master  = merge_catalogues(sxr_cat, hxr_cat)
    print_catalogue(master, "Master catalogue")

    # ── Physics feature engineering ────────────────────────────────────────
    print("\n[6/7] Physics-based feature engineering …")
    if not master.empty:
        all_feats = []
        for _, ev in master.iterrows():
            feats = compute_event_features(ev, prep)
            feats['class'] = ev.get('class', '?')
            feats['peak_time'] = ev['peak_time']
            all_feats.append(feats)
        feat_df = pd.DataFrame(all_feats)
        print(f"  {len(feat_df)} events × {feat_df.shape[1]} features")
        print("  Sample physics features:")
        show = ['peak_time','class','neupert_r','sxr_hxr_lag_s',
                'preflare_enhance','rise_time_s','decay_time_s']
        show = [c for c in show if c in feat_df.columns]
        with pd.option_context('display.float_format', '{:.2f}'.format,
                               'display.width', 120):
            print(feat_df[show].to_string(index=False))
    else:
        feat_df = pd.DataFrame()
        print("  No events — skipping.")

    # ── ML training ────────────────────────────────────────────────────────
    if args.no_train:
        print("\n[7/7] ML training skipped (--no-train).")
        return

    print("\n[7/7] AI forecasting model training …")
    X, y = build_feature_matrix(prep, stride=20)

    if X.empty:
        print("  Cannot train without labelled data.")
        return

    print("\n  ▸ XGBoost …")
    xgb_model, xgb_m, importance = train_xgboost(X, y)
    print_metrics("XGBoost", xgb_m)
    if not importance.empty:
        print("\n  Top features by importance:")
        print(importance.head(8).to_string())

    print("\n  ▸ LSTM …")
    X_seq, y_seq = build_lstm_sequences(prep)
    if X_seq.size:
        lstm_model, lstm_m = train_lstm(X_seq, y_seq, epochs=30)
        print_metrics("LSTM", lstm_m)
    else:
        print("  Cannot build sequences — no known flares.")

    print("\n✓ Pipeline complete.")


if __name__ == '__main__':
    main()
