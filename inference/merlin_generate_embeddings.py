"""
Generate Merlin image embeddings for a set of CT subjects.

Usage:
    python run_merlin_embeddings.py

Before running:
  - Edit INPUT_CSV, SID_COL, PATH_COL, and OUT_DIR below
  - CSV must have one column with subject IDs and one column with
    the full path to each subject's CT scan
"""

import json
import os

import pandas as pd
import torch
from merlin import Merlin
from merlin.data import DataLoader

# ---- 1. Point this at your input CSV -----------------------------------
INPUT_CSV = "/home/km2347/Merlin/test_data/test_inputs.csv"
SID_COL = "SID"
PATH_COL = "Path"  # column name holding full path to each CT scan (.nii.gz)
OUT_DIR = "/home/km2347/Merlin/results"  # where cache + embeddings get written

df = pd.read_csv(INPUT_CSV)

missing_cols = [c for c in (SID_COL, PATH_COL) if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Column(s) {missing_cols} not found in {INPUT_CSV}. "
        f"Available columns: {list(df.columns)}"
    )

# Drop rows with missing path/SID, and check the files actually exist
df = df.dropna(subset=[SID_COL, PATH_COL]).copy()
df["_exists"] = df[PATH_COL].apply(os.path.exists)
missing = df[~df["_exists"]]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} path(s) from the CSV do not exist on disk and will be skipped:")
    for _, row in missing.iterrows():
        print(f"  {row[SID_COL]}: {row[PATH_COL]}")
df = df[df["_exists"]].drop(columns="_exists")

if len(df) == 0:
    raise FileNotFoundError("No valid CT scan paths found — check INPUT_CSV, SID_COL, and PATH_COL.")

subject_ids = df[SID_COL].astype(str).tolist()
ct_paths = df[PATH_COL].tolist()

print(f"Found {len(ct_paths)} valid scans out of {len(pd.read_csv(INPUT_CSV))} rows in CSV.")

# ---- 2. Load model (image-embedding-only mode) ------------------------
model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

# ---- 3. Build datalist + dataloader ------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
datalist = [{"image": p} for p in ct_paths]

dataloader = DataLoader(
    datalist=datalist,
    cache_dir=os.path.join(OUT_DIR, "_merlin_cache"),
    batchsize=2,        # lower this if you hit GPU OOM
    shuffle=False,       # keep order stable so embeddings line up with subject_ids
    num_workers=2,
)

# ---- 4. Run inference ---------------------------------------------------
all_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        emb = model(batch["image"].cuda())
        all_embeddings.append(emb.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)  # (n_subjects, embedding_dim)

# Flatten to (num_samples, embed_dim) regardless of batch structure
if all_embeddings.dim() == 3:
    all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])

print(f"Embeddings shape: {tuple(all_embeddings.shape)}")

# ---- 5. Save results ------------------------------------------------------
import numpy as np

out_path = os.path.join(OUT_DIR, "merlin_embeddings.npz")
np.savez(
    out_path,
    embeddings=all_embeddings.numpy(),
    subject_ids=np.array(subject_ids),
)
print(f"Saved embeddings + subject IDs to {out_path}")

# Also save a CSV mapping SID -> Embedding (JSON-encoded list), matching
# the format used by grab_previous_embeddings.py's output CSV
embeddings_np = all_embeddings.numpy()
embeddings_df = pd.DataFrame({
    SID_COL: subject_ids,
    "Embedding": [json.dumps(row.tolist()) for row in embeddings_np],
})
csv_out_path = os.path.join(OUT_DIR, "merlin_embeddings.csv")
embeddings_df.to_csv(csv_out_path, index=False)
print(f"Saved embeddings CSV to {csv_out_path}")