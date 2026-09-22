"""
Generate Merlin image embeddings for a set of CT subjects.

Usage:
    python run_merlin_embeddings.py

Before running:
  - pip install merlin-vlm
  - Edit CT_DIR below to point at your data
  - Your scans must be NIfTI (.nii.gz)
"""

import glob
import os

import torch
from merlin import Merlin
from merlin.data import DataLoader

# ---- 1. Point this at your CT scans ----------------------------------
# Adjust the glob pattern to match how your files are actually laid out.
# Example assumes: /path/to/your/ct_data/<subject_id>/image.nii.gz
CT_DIR = "/path/to/your/ct_data"
ct_paths = sorted(glob.glob(os.path.join(CT_DIR, "*", "image.nii.gz")))

if not ct_paths:
    raise FileNotFoundError(f"No .nii.gz files found under {CT_DIR} — check your path/pattern.")

# Keep subject IDs aligned with ct_paths (assumes parent folder name = subject id)
subject_ids = [os.path.basename(os.path.dirname(p)) for p in ct_paths]

print(f"Found {len(ct_paths)} scans.")

# ---- 2. Load model (image-embedding-only mode) ------------------------
model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

# ---- 3. Build datalist + dataloader ------------------------------------
datalist = [{"image": p} for p in ct_paths]

dataloader = DataLoader(
    datalist=datalist,
    cache_dir=os.path.join(CT_DIR, "_merlin_cache"),
    batchsize=4,        # lower this if you hit GPU OOM
    shuffle=False,       # keep order stable so embeddings line up with subject_ids
    num_workers=4,
)

# ---- 4. Run inference ---------------------------------------------------
all_embeddings = []
with torch.no_grad():
    for batch in dataloader:
        emb = model(batch["image"].cuda())
        all_embeddings.append(emb.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0)  # (n_subjects, embedding_dim)
print(f"Embeddings shape: {tuple(all_embeddings.shape)}")

# ---- 5. Save results ------------------------------------------------------
import numpy as np

out_path = os.path.join(CT_DIR, "merlin_embeddings.npz")
np.savez(
    out_path,
    embeddings=all_embeddings.numpy(),
    subject_ids=np.array(subject_ids),
)
print(f"Saved embeddings + subject IDs to {out_path}")