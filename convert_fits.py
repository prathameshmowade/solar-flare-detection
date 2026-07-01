#!/usr/bin/env python3
"""
convert_fits.py — Aditya-L1 SoLEXS / HEL1OS FITS → JSON
for the Solar Flare Monitor browser prototype.

Requirements:
    pip install astropy numpy

Quick start:
    # 1. Inspect your FITS file first to see column names:
    python convert_fits.py --inspect solexs_l1.fits

    # 2. Convert both instruments to one JSON:
    python convert_fits.py --solexs solexs_l1.fits --helios helios_l1.fits

    # 3. Upload the output JSON in the browser app (solar_flare_monitor.html).

If auto-detection fails, specify columns manually:
    python convert_fits.py --solexs s.fits --time-col TIME --flux-cols RATE_1,RATE_2,RATE_3
"""

import json
import sys
import os
import argparse
import numpy as np


# ── Inspect ───────────────────────────────────────────────────────────────────
def inspect(filepath: str) -> None:
    """Print the complete FITS structure so you can identify column names."""
    from astropy.io import fits

    print(f"\n{'═' * 55}")
    print(f"  FITS: {filepath}")
    print(f"{'═' * 55}")

    with fits.open(filepath) as hdul:
        hdul.info()
        for i, hdu in enumerate(hdul):
            cols = getattr(hdu, "columns", None)
            if cols:
                print(f"\nExtension {i}  ({hdu.name})  — {len(hdul[i].data)} rows")
                print(f"  {'Column':<30} {'Format':<12} {'Unit'}")
                print(f"  {'-'*30} {'-'*12} {'-'*10}")
                for c in cols:
                    print(f"  {c.name:<30} {c.format:<12} {c.unit or ''}")

    print(f"\n{'─' * 55}")
    print("Tip: once you know the column names, run:")
    print(f"  python convert_fits.py --solexs {filepath} \\")
    print(f"      --time-col TIME --flux-cols COL1,COL2,COL3")


# ── Read one instrument ───────────────────────────────────────────────────────
def read_instrument(
    filepath: str,
    time_col: str | None = None,
    flux_cols: list[str] | None = None,
    ext: int = 1,
) -> dict | None:
    """
    Read a FITS instrument file; return dict with 'time' and 'channels'.

    Returns None on failure.
    """
    from astropy.io import fits

    with fits.open(filepath) as hdul:
        # Try the requested extension; fall back if empty
        data = None
        for candidate in [ext, 1, 2, 0]:
            try:
                d = hdul[candidate].data
                if d is not None and len(d) > 0:
                    data = d
                    if candidate != ext:
                        print(f"  (fell back to extension {candidate})")
                    break
            except Exception:
                continue

        if data is None:
            print("  ✗ No data found in FITS file.")
            return None

        col_names = [c.name for c in data.columns] if hasattr(data, "columns") else []
        if not col_names:
            print("  ✗ No column metadata found.")
            return None

        # ── Auto-detect time column ──────────────────────────────────────────
        if time_col is None:
            candidates = ["TIME", "Time", "time", "T", "UTC_TIME", "MET", "TSTART", "EPOCH"]
            for name in candidates:
                if name in col_names:
                    time_col = name
                    break
            if time_col is None:
                for name in col_names:
                    if "TIME" in name.upper():
                        time_col = name
                        break
            if time_col is None:
                print(f"  ✗ Could not find a time column. Available: {col_names}")
                print("    Use --time-col to specify one.")
                return None

        print(f"  Time column  : {time_col}")
        t = np.array(data[time_col], dtype=float)
        t -= t[0]   # normalise to start at 0

        # ── Auto-detect flux columns ─────────────────────────────────────────
        if flux_cols is None:
            keywords = {"RATE", "FLUX", "COUNT", "CTS", "COUNTS", "DATA", "CHANNEL", "CH"}
            flux_cols = []
            for name in col_names:
                if name == time_col:
                    continue
                try:
                    arr = np.array(data[name], dtype=float)
                    if arr.ndim == 1 and len(arr) == len(t):
                        is_flux = (
                            any(kw in name.upper() for kw in keywords)
                            or name.startswith("CH")
                            or name.startswith("E_")
                        )
                        if is_flux:
                            flux_cols.append(name)
                except Exception:
                    pass

            # Fallback: all remaining 1-D numeric columns
            if not flux_cols:
                for name in col_names:
                    if name == time_col:
                        continue
                    try:
                        arr = np.array(data[name], dtype=float)
                        if arr.ndim == 1 and len(arr) == len(t):
                            flux_cols.append(name)
                    except Exception:
                        pass

        if not flux_cols:
            print(f"  ✗ No flux columns found. Available: {col_names}")
            return None

        print(f"  Flux columns : {flux_cols}")

        # ── Extract and clean channel data ───────────────────────────────────
        channels: dict[str, list] = {}
        for name in flux_cols:
            try:
                arr = np.array(data[name], dtype=float)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                arr = np.clip(arr, 0, None)      # no negative counts
                channels[name] = arr.tolist()
            except Exception as exc:
                print(f"  Warning: skipping column '{name}' ({exc})")

        if not channels:
            print("  ✗ No valid channel data extracted.")
            return None

        return {"time": t.tolist(), "channels": channels}


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert_fits.py",
        description="Convert Aditya-L1 SoLEXS / HEL1OS FITS → JSON "
                    "for the Solar Flare Monitor browser app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python convert_fits.py --inspect solexs_l1.fits
  python convert_fits.py --solexs solexs_l1.fits --helios helios_l1.fits
  python convert_fits.py --solexs s.fits --time-col TIME --flux-cols RATE_0,RATE_1,RATE_2
  python convert_fits.py --solexs s.fits --helios h.fits -o my_event.json
        """,
    )
    parser.add_argument("--inspect", metavar="FILE",
                        help="print FITS structure and exit (no conversion)")
    parser.add_argument("--solexs", metavar="FILE", help="SoLEXS Level-1 FITS file")
    parser.add_argument("--helios", metavar="FILE", help="HEL1OS Level-1 FITS file")
    parser.add_argument("--time-col", default=None,
                        help="column name for time (auto-detected if omitted)")
    parser.add_argument("--flux-cols", default=None,
                        help="comma-separated list of flux column names (auto-detected if omitted)")
    parser.add_argument("--ext", type=int, default=1,
                        help="FITS extension number to read (default: 1)")
    parser.add_argument("-o", "--output", default="aditya_l1_data.json",
                        help="output JSON path (default: aditya_l1_data.json)")
    args = parser.parse_args()

    # Dependency check
    try:
        from astropy.io import fits  # noqa: F401
    except ImportError:
        print("astropy is required:  pip install astropy numpy")
        sys.exit(1)

    # Inspect-only mode
    if args.inspect:
        inspect(args.inspect)
        return

    if not args.solexs and not args.helios:
        parser.print_help()
        print("\nError: supply at least one of --solexs or --helios.")
        sys.exit(1)

    flux_cols = args.flux_cols.split(",") if args.flux_cols else None

    result: dict = {
        "mission":        "Aditya-L1",
        "format_version": "1.0",
        "solexs":         None,
        "helios":         None,
    }

    if args.solexs:
        print(f"\nReading SoLEXS: {args.solexs}")
        result["solexs"] = read_instrument(args.solexs, args.time_col, flux_cols, args.ext)
        if result["solexs"]:
            n  = len(result["solexs"]["time"])
            ch = list(result["solexs"]["channels"].keys())
            print(f"  ✓ {n} time points,  {len(ch)} channel(s): {ch}")

    if args.helios:
        print(f"\nReading HEL1OS: {args.helios}")
        result["helios"] = read_instrument(args.helios, args.time_col, flux_cols, args.ext)
        if result["helios"]:
            n  = len(result["helios"]["time"])
            ch = list(result["helios"]["channels"].keys())
            print(f"  ✓ {n} time points,  {len(ch)} channel(s): {ch}")

    # Save
    with open(args.output, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n  Saved → {args.output}  ({size_kb:.1f} KB)")
    print("  Upload this file in the Solar Flare Monitor app.")


if __name__ == "__main__":
    main()
