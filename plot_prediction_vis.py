"""
Prediction Visualization — replicate and plot `ds_image_data` from quick_start.ipynb.

Produces a single 3x5 grid:
    - 3 Rows (Row Titles): DRIVE, CHASE-DB1, STARE (rotated along y-axis)
    - 5 Columns (Column Titles): Original Image, Ground Truth, Finetuned SAM, Med-SA, Ours
    - All images in each row are resized to match the shape of the row's first image.
"""

from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_image_grids(
    img_grids,
    titles=None,
    row_titles=None,
    col_titles=None,
    figsize_per_image=3,
    save_path=None
):
    """
    Plot image grids with column titles and vertical row titles aligned to the y-axis.

    Parameters
    ----------
    img_grids : list or ndarray
        Either a list of m×n image grids [grid1, grid2, ...],
        or a single m×n grid (2D list/array of numpy images).
    titles : list[str], optional
        Titles centered above each grid.
    row_titles : list[str], optional
        One title per row, rotated along the y-axis on the left.
    col_titles : list[str], optional
        One title per column, displayed at the top of each column.
    figsize_per_image : float
        Size in inches per image cell.
    save_path : str or Path, optional
        Path to save the generated figure.
    """
    if isinstance(img_grids, list) and len(img_grids) > 0 and isinstance(img_grids[0][0], np.ndarray):
        grid_matrix = img_grids
        m = len(grid_matrix)
        n = len(grid_matrix[0])
        is_single_matrix = True
    elif isinstance(img_grids, list) and len(img_grids) > 0 and isinstance(img_grids[0][0], list):
        is_single_matrix = False
        num_grids = len(img_grids)
        m = len(img_grids[0])
        n = len(img_grids[0][0])
        total_cols = num_grids * n
    else:
        grid_matrix = img_grids
        m = len(grid_matrix)
        n = len(grid_matrix[0])
        is_single_matrix = True

    if is_single_matrix:
        total_cols = n
        fig, axes = plt.subplots(
            m,
            total_cols,
            figsize=(figsize_per_image * total_cols, figsize_per_image * m),
            squeeze=False
        )

        for r in range(m):
            for c in range(n):
                ax = axes[r, c]
                img = grid_matrix[r][c]

                if img is not None and img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if img is not None:
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

                # Keep axis frame hidden but allow y-axis label to display
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # Set column titles at top
        if col_titles:
            for c in range(min(n, len(col_titles))):
                axes[0, c].set_title(col_titles[c], fontsize=16, weight="bold", pad=12)

        # Set row titles on left rotated aligned to the y-axis
        if row_titles:
            for r in range(min(m, len(row_titles))):
                axes[r, 0].set_ylabel(
                    row_titles[r],
                    fontsize=16,
                    weight="bold",
                    labelpad=15,
                    rotation=90,
                    va="center",
                    ha="center"
                )

        if titles and len(titles) > 0 and titles[0]:
            fig.suptitle(titles[0], fontsize=18, weight="bold", y=1.02)

    else:
        fig, axes = plt.subplots(
            m,
            total_cols,
            figsize=(figsize_per_image * total_cols, figsize_per_image * m),
            squeeze=False
        )

        if titles is None:
            titles = [None] * num_grids

        for g, grid in enumerate(img_grids):
            col_offset = g * n

            for r in range(m):
                for c in range(n):
                    ax = axes[r, col_offset + c]
                    img = grid[r][c]

                    if img is not None and img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    if img is not None:
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)

            if titles[g]:
                center_col = col_offset + (n - 1) / 2
                x = (center_col + 0.5) / total_cols
                fig.text(
                    x, 0.98, titles[g],
                    ha="center", va="top",
                    fontsize=16, weight="bold"
                )

        if col_titles:
            for c in range(min(total_cols, len(col_titles))):
                axes[0, c].set_title(col_titles[c], fontsize=14, weight="bold", pad=10)

        if row_titles:
            for r in range(min(m, len(row_titles))):
                axes[r, 0].set_ylabel(
                    row_titles[r],
                    fontsize=16,
                    weight="bold",
                    labelpad=15,
                    rotation=90,
                    va="center",
                    ha="center"
                )

    plt.tight_layout(rect=[0.02, 0.0, 1.0, 0.95] if row_titles else [0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.close(fig)


# ---------- Load prediction vis image data ----------
aug_sample_prefixes = ("random", "h", "v")
dataset_dirs = {
    "DRIVE": "data/drive_aug",
    "CHASE-DB1": "data/chase_aug",
    "STARE": "data/stare_aug",
}

pred_dirs = [dir_ / "Samples" for dir_ in Path("logs").glob("vis*/")]

ds_image_data = {}
random.seed(24)

for ds_name, ds_dir in dataset_dirs.items():
    image_dir = Path(ds_dir) / "test/images"
    label_dir = Path(ds_dir) / "test/labels"

    ds_pred_dirs = sorted([dir_ for dir_ in pred_dirs if ds_name.split("-")[0].lower() in str(dir_)])
    sample_image_stem = random.choice([path.stem for path in ds_pred_dirs[0].iterdir()])

    ds_image_data[ds_name] = [
        [path for path in dir_.iterdir() if path.stem.startswith(sample_image_stem)][0]
        for dir_ in [
            image_dir,
            label_dir,
            ds_pred_dirs[2],
            ds_pred_dirs[0],
            ds_pred_dirs[1],
        ]
    ]

# Construct 3x5 grid matrix:
# Rows: DRIVE, CHASE-DB1, STARE
# Columns: Original Image, Ground Truth, Finetuned SAM, Med-SA, Ours
row_titles = ["DRIVE", "CHASE-DB1", "STARE"]
col_titles = ["Original Image", "Ground Truth", "Finetuned SAM", "Med-SA", "Ours"]

grid_matrix = []
for ds_name in row_titles:
    img_paths = ds_image_data[ds_name]
    
    # Load first image (Original Image) to establish row target shape
    first_img = cv2.imread(str(img_paths[0]))
    target_h, target_w = first_img.shape[:2]
    
    row_imgs = [first_img]
    for p in img_paths[1:]:
        img = cv2.imread(str(p))
        if img is not None and (img.shape[0] != target_h or img.shape[1] != target_w):
            interp = cv2.INTER_AREA if (img.shape[0] > target_h or img.shape[1] > target_w) else cv2.INTER_CUBIC
            img = cv2.resize(img, (target_w, target_h), interpolation=interp)
        row_imgs.append(img)
        
    grid_matrix.append(row_imgs)

# Plot and save
save_path = Path("prediction_vis.png")
plot_image_grids(
    grid_matrix,
    row_titles=row_titles,
    col_titles=col_titles,
    figsize_per_image=3.5,
    save_path=save_path
)
print("Prediction Vis with per-row image resizing completed successfully.")
