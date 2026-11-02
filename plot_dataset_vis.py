"""
Dataset Visualization — replicate the "Dataset Vis" cells from quick_start.ipynb.

Produces a single figure with 3 sub-images (one per dataset: DRIVE, CHASE-DB1,
STARE).  Each sub-image is a 2×2 grid:
    row 0: two sample *images*
    row 1: the corresponding *labels*
"""

from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend so it works headless
import matplotlib.pyplot as plt


# ---------- plot_image_grids (copied from the notebook) ----------
def plot_image_grids(img_grids, titles=None, figsize_per_image=3, save_path=None):
    """
    Plot multiple m×n image grids side-by-side.

    Parameters
    ----------
    img_grids : list
        List of image grids.
        Each grid is a nested list (or ndarray) of shape (m, n),
        where each element is a numpy image.
    titles : list[str], optional
        One title per grid.
    figsize_per_image : float
        Size (inches) allocated to each image.
    save_path : str or Path, optional
        If given, save the figure to this path.
    """

    num_grids = len(img_grids)
    m = len(img_grids[0])
    n = len(img_grids[0][0])

    total_cols = num_grids * n

    fig, axes = plt.subplots(
        m,
        total_cols,
        figsize=(figsize_per_image * total_cols,
                 figsize_per_image * m)
    )

    if m == 1:
        axes = np.expand_dims(axes, axis=0)

    if titles is None:
        titles = [None] * num_grids

    for g, grid in enumerate(img_grids):
        col_offset = g * n

        # Draw images
        for r in range(m):
            for c in range(n):
                ax = axes[r, col_offset + c]

                img = grid[r][c]

                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                ax.axis("off")

        # Center title above this grid
        if titles[g]:
            center_col = col_offset + (n - 1) / 2
            x = (center_col + 0.5) / total_cols

            fig.text(
                x,
                0.98,
                titles[g],
                ha="center",
                va="top",
                fontsize=16,
                weight="bold"
            )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.close(fig)


# ---------- Dataset sampling (adapted from the notebook) ----------
aug_sample_prefixes = ("random", "h", "v")

dataset_dirs = {
    "DRIVE":    "data/drive_aug",
    "CHASE-DB1": "data/chase_aug",
    "STARE":    "data/stare_aug",
}
sample_n = 2
sample_dataset_image_labels = {}

for ds_name, ds_dir in dataset_dirs.items():
    image_dir = Path(ds_dir) / "test/images"
    image_paths = [
        path for path in image_dir.glob("*.*")
        if not str(path.name).startswith(aug_sample_prefixes)
    ]
    sample_image_paths = random.sample(image_paths, sample_n)

    sample_dataset_image_labels[ds_name] = list(sample_image_paths)

    label_dir = Path(ds_dir) / "test/labels"
    label_paths = list(label_dir.glob("*.*"))

    for sample_image_path in sample_image_paths:
        sample_id = (
            "_".join(sample_image_path.stem.split("_")[:-1])
            if "_" in sample_image_path.stem
            else sample_image_path.stem
        )
        sample_label_path = [
            path for path in label_paths if path.name.startswith(sample_id)
        ][0]
        sample_dataset_image_labels[ds_name].append(sample_label_path)


# ---------- Build 2×2 grids and plot ----------
# Each value in sample_dataset_image_labels is [img0, img1, label0, label1].
# We reshape that into a 2×2 grid:
#   row 0 = [img0,   img1  ]
#   row 1 = [label0, label1]

image_grids = []
for ds_image_labels in sample_dataset_image_labels.values():
    # ds_image_labels = [img_path0, img_path1, label_path0, label_path1]
    imgs = [cv2.imread(str(p)) for p in ds_image_labels]
    grid = [
        [imgs[0], imgs[1]],     # row 0: images
        [imgs[2], imgs[3]],     # row 1: labels
    ]
    image_grids.append(grid)

titles = list(sample_dataset_image_labels.keys())

save_path = Path("dataset_vis.png")
plot_image_grids(image_grids, titles, save_path=save_path)
print("Done.")
