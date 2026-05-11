import matplotlib
matplotlib.use("QtAgg")
import os
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from glob import glob
import torch

from sam2.build_sam import build_sam2_video_predictor

# =========================
# CONFIG
# =========================
IMAGE_DIR = "data/sam2/images"

OUT_MASK_DIR = "data/sam2/final_masks"
OUT_OVERLAY_DIR = "data/sam2/overlays"
TEMP_FRAME_DIR = "data/sam2/sam2_temp_frames"
CSV_PATH = "data/sam2/droplet_area_timeseries_sam2.csv"

# Update these paths
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

PIXEL_SIZE = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBJ_ID = 1
MIN_COMPONENT_AREA = 200

os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_OVERLAY_DIR, exist_ok=True)
os.makedirs(TEMP_FRAME_DIR, exist_ok=True)

# =========================
# IMAGE LOADING / CONVERSION
# =========================
def load_image_gray(path):
    img = tiff.imread(path)

    if img.ndim > 2:
        img = img[..., 0]

    img = img.astype(np.float32)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_uint8 = img_norm.astype(np.uint8)

    return img_uint8


def prepare_sam2_frame_folder(image_paths):
    """
    SAM2 video predictor expects a folder of video frames.
    We convert your TIFF/PNG series into numbered JPG frames.
    """

    frame_paths = []

    for i, path in enumerate(image_paths):
        gray = load_image_gray(path)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        frame_name = f"{i:05d}.jpg"
        frame_path = os.path.join(TEMP_FRAME_DIR, frame_name)

        cv2.imwrite(frame_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)

    return frame_paths


# =========================
# LIVE FIRST-FRAME ANNOTATION
# =========================
def get_first_frame_mask_live_sam2(gray, predictor, inference_state):
    """
    Live first-frame SAM2 annotation.

    Left click   = positive point
    Middle click = negative point
    Right click  = undo last point
    Enter        = accept current mask
    """

    points = []
    labels = []

    current_mask = None
    current_logits = None
    mask_artist = None
    point_artists = []

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(gray, cmap="gray")
    ax.set_title(
        "Left = droplet | Middle = background/electrode | Right = undo | Enter = accept"
    )

    def redraw_points():
        nonlocal point_artists

        for artist in point_artists:
            artist.remove()

        point_artists = []

        for (x, y), lab in zip(points, labels):
            if lab == 1:
                artist = ax.plot(x, y, "go", markersize=7)[0]
            else:
                artist = ax.plot(x, y, "ro", markersize=7)[0]

            point_artists.append(artist)

    def update_prediction():
        nonlocal current_mask, current_logits, mask_artist

        if len(points) == 0:
            return

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        with torch.inference_mode():
            if DEVICE == "cuda":
                autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
            else:
                autocast_context = torch.autocast("cpu", enabled=False)

            with autocast_context:
                _, _, mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=OBJ_ID,
                    points=point_coords,
                    labels=point_labels,
                )

        current_logits = mask_logits[0]
        current_mask = (mask_logits[0] > 0).cpu().numpy()
        current_mask = np.squeeze(current_mask).astype(bool)

        if mask_artist is not None:
            mask_artist.remove()

        overlay = np.zeros((*current_mask.shape, 4), dtype=np.float32)
        overlay[current_mask] = [1.0, 0.35, 0.0, 0.45]

        mask_artist = ax.imshow(overlay)
        redraw_points()
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        # left click = positive
        if event.button == 1:
            points.append([x, y])
            labels.append(1)

        # middle click = negative
        elif event.button == 2:
            points.append([x, y])
            labels.append(0)

        # right click = undo
        elif event.button == 3:
            if len(points) > 0:
                points.pop()
                labels.pop()

                if len(points) == 0:
                    nonlocal_clear_mask()
                    fig.canvas.draw_idle()
                    return
            else:
                return

        update_prediction()

    def nonlocal_clear_mask():
        nonlocal current_mask, current_logits, mask_artist

        current_mask = None
        current_logits = None

        if mask_artist is not None:
            mask_artist.remove()
            mask_artist = None

        redraw_points()

    def onkey(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show(block=True)

    if current_mask is None:
        raise RuntimeError("No SAM2 mask generated. Add at least one positive point.")

    return current_mask, current_logits, np.array(points), np.array(labels)


def preview_first_mask(gray, mask):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(gray, cmap="gray")

    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.35, 0.0, 0.45]

    ax.imshow(overlay)
    ax.set_title("SAM2 first-frame mask preview. Close window to continue.")
    plt.show()


# =========================
# MASK POST-PROCESSING
# =========================
def keep_reasonable_components(mask):
    mask_uint8 = mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, 8)

    cleaned = np.zeros_like(mask_uint8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 1

    return cleaned.astype(bool)


def merge_electrode_split_mask(mask):
    """
    Reconstruct whole droplet from visible SAM2 mask.
    Convex hull bridges electrode-split regions.
    """

    mask = keep_reasonable_components(mask)
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return mask.astype(bool)

    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)

    filled = np.zeros_like(mask_uint8)
    cv2.drawContours(filled, [hull], -1, 1, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel)

    return filled.astype(bool)


def save_overlay(gray, mask, out_path):
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    colored = overlay.copy()
    colored[mask] = (
        0.6 * colored[mask] + 0.4 * np.array([0, 120, 255])
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(colored, [cnt], -1, (0, 255, 255), 2)

    cv2.imwrite(out_path, colored)


# =========================
# MAIN SAM2 PIPELINE
# =========================
def main():
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*")))

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    print(f"Found {len(image_paths)} images.")
    print("Preparing SAM2 frame folder...")
    prepare_sam2_frame_folder(image_paths)

    print("Loading SAM2...")
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=DEVICE
    )

    print("Initializing SAM2 video state...")
    inference_state = predictor.init_state(video_path=TEMP_FRAME_DIR)

    first_gray = load_image_gray(image_paths[0])

    print("Annotate first frame:")
    print("  Left click   = positive droplet point")
    print("  Middle click = negative electrode/background point")
    print("  Enter        = accept points")

    first_mask, first_logits, points, labels = get_first_frame_mask_live_sam2(
        first_gray,
        predictor,
        inference_state
    )


    print("Propagating mask through image series...")

    frame_masks = {}

    with torch.inference_mode():
        if DEVICE == "cuda":
            autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_context = torch.autocast("cpu", enabled=False)

        with autocast_context:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state
            ):
                mask = (out_mask_logits[0] > 0).cpu().numpy()
                mask = np.squeeze(mask).astype(bool)
                frame_masks[out_frame_idx] = mask

    results = []

    for frame_idx, path in enumerate(image_paths):
        fname = os.path.basename(path)
        gray = load_image_gray(path)

        if frame_idx not in frame_masks:
            print(f"Warning: no SAM2 mask for frame {frame_idx}. Saving empty mask.")
            sam2_mask = np.zeros_like(gray, dtype=bool)
        else:
            sam2_mask = frame_masks[frame_idx]

        final_mask = merge_electrode_split_mask(sam2_mask)

        area_pixels = int(final_mask.sum())
        area_physical = area_pixels * (PIXEL_SIZE ** 2)

        mask_out = os.path.join(OUT_MASK_DIR, fname)
        overlay_out = os.path.join(OUT_OVERLAY_DIR, fname)

        tiff.imwrite(mask_out, final_mask.astype(np.uint8), compression=None)
        save_overlay(gray, final_mask, overlay_out)

        results.append({
            "frame": frame_idx,
            "image": fname,
            "area_pixels": area_pixels,
            "area_physical": area_physical,
            "mask_path": mask_out,
            "overlay_path": overlay_out
        })

        print(f"{fname}: area = {area_pixels} pixels")

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)

    print("\nDone.")
    print(f"Masks saved to: {OUT_MASK_DIR}")
    print(f"Overlays saved to: {OUT_OVERLAY_DIR}")
    print(f"CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()