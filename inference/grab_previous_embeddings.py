#!/usr/bin/env python3
"""Build a CSV (SID, Embedding) from a folder of previously-extracted
TANGERINE .npz embeddings, restricted to a specific subject cohort.

Hardcoded for this one setup -- edit the CONFIG block below if any path or
column name changes. No command-line arguments.

Steps:
  1. Load TARGET_SIDS_CSV and collect the set of target SIDs.
  2. Scan NPZ_DIR for files matching NPZ_GLOB (optionally filtered to one
     SCAN_TYPE, using the INSP/EXP filename-substring convention), and
     restrict to the target SIDs.
  3. Load each .npz's NPZ_KEY array, squeeze to 1D, skip anything malformed.
  4. Save SID + Embedding (JSON-encoded list) to OUTPUT_CSV.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- edit these if a path or column name changes
# --------------------------------------------------------------------------

TARGET_SIDS_CSV = "/home/km2347/Merlin/test_data/test_inputs.csv"
TARGET_SID_COL = "SID"

NPZ_DIR = "/home/ahc44/Datos/COPDGene/COPDGeneEmbeddingFM/"
NPZ_GLOB = "*merlin*embedding*.npz"
NPZ_KEY = "image"
SCAN_TYPE = "INSP"                # "INSP", "EXP", or None to disable filtering

SID_COL_OUT = "SID"
EMBEDDING_COL_OUT = "Embedding"
OUTPUT_CSV = "previous_merlin_embeddings.csv"

# --------------------------------------------------------------------------
# Step 1: load target SIDs (the 168 subjects)
# --------------------------------------------------------------------------

print(f"Loading target SIDs from {TARGET_SIDS_CSV}...")
df_targets = pd.read_csv(TARGET_SIDS_CSV)
target_sids = set(df_targets[TARGET_SID_COL].astype(str))
print(f"Loaded {len(target_sids)} target SIDs")

# --------------------------------------------------------------------------
# Step 2: find + filter .npz files
# --------------------------------------------------------------------------

print(f"\nScanning {NPZ_DIR} for '{NPZ_GLOB}'...")
npz_paths = sorted(Path(NPZ_DIR).rglob(NPZ_GLOB))

if SCAN_TYPE:
    other = "EXP" if SCAN_TYPE == "INSP" else "INSP"
    n_before = len(npz_paths)
    npz_paths = [p for p in npz_paths if SCAN_TYPE in p.name and other not in p.name]
    print(f"scan_type={SCAN_TYPE}: kept {len(npz_paths)}/{n_before} files")
else:
    print(f"{len(npz_paths)} files found (no scan_type filtering)")

n_before_target_filter = len(npz_paths)
npz_paths = [p for p in npz_paths if p.stem.split("_")[0] in target_sids]
print(f"target SIDs: kept {len(npz_paths)}/{n_before_target_filter} files")

# --------------------------------------------------------------------------
# Step 3: load embeddings
# --------------------------------------------------------------------------

rows = []
n_skipped_wrong_key = n_skipped_error = 0
for p in npz_paths:
    sid = p.stem.split("_")[0]
    try:
        data = np.load(p, allow_pickle=True)
        if NPZ_KEY not in data.files:
            n_skipped_wrong_key += 1
            continue
        embedding = np.asarray(data[NPZ_KEY]).squeeze()
        if embedding.ndim != 1:
            n_skipped_wrong_key += 1
            continue
    except Exception as exc:
        n_skipped_error += 1
        print(f"  Failed to load {p.name}: {exc}")
        continue
    rows.append({SID_COL_OUT: sid, EMBEDDING_COL_OUT: json.dumps(embedding.tolist())})

df_emb = pd.DataFrame(rows, columns=[SID_COL_OUT, EMBEDDING_COL_OUT])
print(
    f"\nLoaded {len(df_emb)} embeddings "
    f"({n_skipped_wrong_key} skipped: wrong key/shape | {n_skipped_error} skipped: load error)"
)
if df_emb["SID"].duplicated().any() if len(df_emb) else False:
    n_dupes = int(df_emb["SID"].duplicated().sum())
    print(f"  WARNING: {n_dupes} duplicate SIDs among loaded embeddings "
          f"(e.g. both INSP and EXP matched, or repeat scans) -- keeping the first occurrence.")
    df_emb = df_emb.drop_duplicates(subset=SID_COL_OUT, keep="first")

missing_sids = target_sids - set(df_emb["SID"])
if missing_sids:
    print(f"  WARNING: {len(missing_sids)} target SIDs had no matching .npz, e.g.: {list(missing_sids)[:5]}")

# --------------------------------------------------------------------------
# Step 4: save
# --------------------------------------------------------------------------

df_emb.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df_emb)} rows ({SID_COL_OUT}, {EMBEDDING_COL_OUT}) to {OUTPUT_CSV}")