"""
make_loss_animation.py

Usage:
    python make_loss_animation.py path/to/train.log

Produces:
    - /mnt/data/loss_animation.gif
    - /mnt/data/loss_animation.mp4
    - ./losses.csv  (parsed step,loss)
"""

import re
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# ---------- CONFIG ----------
OUT_GIF = "loss_animation.gif"
OUT_MP4 = "loss_animation.mp4"
CSV_OUT = "losses.csv"

# Animation parameters you can tweak:
FPS = 20                 # frames per second in the output video/gif
DURATION_S = 12          # total seconds of the animation
SMOOTH_WINDOW = 1        # rolling window for smoothing (set >1 to smooth)
MAX_POINTS_IN_ANIM = 1200  # to keep animation reasonable, subsample if too many
# ---------------------------

def parse_losses_from_text(text):
    """
    Extract floats from patterns:
        loss=0.01234
        Loss: 0.01234
        loss: 0.01234
    Return list of (index, loss) in the order found.
    """
    losses = []

    # Find tokens like loss=0.0123 or loss=0.0123,loss=...
    for m in re.finditer(r'loss=([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE):
        val = float(m.group(1))
        losses.append(val)

    # Find lines like "Loss: 0.0123" or "Loss: 0.0123\n" or "Loss: 0.0123]"
    for m in re.finditer(r'Loss[:\s]+\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE):
        val = float(m.group(1))
        losses.append(val)

    # In case there are also tokens like "loss: 0.0123" (lowercase)
    for m in re.finditer(r'loss[:\s]+\s*([0-9]*\.?[0-9]+)', text, flags=re.IGNORECASE):
        # avoid duplicates (we already caught many via loss=)
        val = float(m.group(1))
        losses.append(val)

    # The above can duplicate captures; keep order but dedupe consecutive identical sequence by index length check.
    # However we want to preserve order; so we'll return the list as-is but it's common to have duplicates.
    return losses

def subsample_for_animation(xs, ys, max_points):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    # pick indices uniformly to keep the timeline shape
    idx = np.linspace(0, n-1, max_points).astype(int)
    return [xs[i] for i in idx], [ys[i] for i in idx]

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_loss_animation.py path/to/train.log")
        sys.exit(1)

    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print("File not found:", log_path)
        sys.exit(1)

    with open(log_path, "r", errors="ignore") as f:
        text = f.read()

    losses = parse_losses_from_text(text)
    if len(losses) == 0:
        print("No loss values found in the log. Patterns looked for: 'loss=0.0123', 'Loss: 0.0123', 'loss: 0.0123'")
        sys.exit(1)

    steps = list(range(1, len(losses)+1))
    df = pd.DataFrame({"step": steps, "loss": losses})
    df.to_csv(CSV_OUT, index=False)
    print(f"Parsed {len(losses)} loss values. Saved to {CSV_OUT}")

    # smoothing (optional)
    if SMOOTH_WINDOW > 1:
        df['loss_smooth'] = df['loss'].rolling(window=SMOOTH_WINDOW, min_periods=1, center=False).mean()
    else:
        df['loss_smooth'] = df['loss']

    xs = df['step'].tolist()
    ys = df['loss_smooth'].tolist()

    # Optionally subsample to limit animation size
    xs_anim, ys_anim = subsample_for_animation(xs, ys, MAX_POINTS_IN_ANIM)

    # Normalize axis ranges
    xmin, xmax = min(xs_anim), max(xs_anim)
    ymin, ymax = min(ys_anim), max(ys_anim)
    yrange = ymax - ymin
    if yrange == 0:
        # avoid degenerate axis
        ymin -= 1e-3
        ymax += 1e-3
    else:
        # add small margins
        ymin -= 0.05 * yrange
        ymax += 0.05 * yrange

    fig, ax = plt.subplots(figsize=(10,4))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Training step (parsed order)")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (animated) — shows fluctuation")

    line, = ax.plot([], [], lw=1)
    point, = ax.plot([], [], marker='o')

    # Optionally show a trailing average text
    avg_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va='top')

    total_frames = max(20, int(FPS * DURATION_S))
    # Map frames 0..total_frames-1 to indices in xs_anim
    def frame_to_idx(frame):
        # linear mapping
        frac = frame / float(total_frames-1)
        idx = int(frac * (len(xs_anim)-1))
        return idx

    def init():
        line.set_data([], [])
        point.set_data([], [])
        avg_text.set_text("")
        return line, point, avg_text

    def update(frame):
        idx = frame_to_idx(frame)
        # show from start to idx
        xdata = xs_anim[:idx+1]
        ydata = ys_anim[:idx+1]
        line.set_data(xdata, ydata)
        point.set_data(xdata[-1:], ydata[-1:])
        # show rolling mean of last N points
        window = max(1, int(len(xdata)*0.02))
        mean_recent = np.mean(ydata[-window:])
        avg_text.set_text(f"step {xdata[-1]}  recent mean ≈ {mean_recent:.5f}")
        return line, point, avg_text

    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=True)

    # Save GIF
    try:
        print("Saving GIF (may take a few seconds)...")
        writer = PillowWriter(fps=FPS)
        anim.save(OUT_GIF, writer=writer)
        print("Saved", OUT_GIF)
    except Exception as e:
        print("Failed to save GIF:", e)

    # Save MP4 (if ffmpeg available)
    try:
        print("Saving MP4 (may take a few seconds)...")
        ff_writer = FFMpegWriter(fps=FPS)
        anim.save(OUT_MP4, writer=ff_writer)
        print("Saved", OUT_MP4)
    except Exception as e:
        print("Failed to save MP4 (ffmpeg may be missing):", e)

    print("Done. Files produced:", CSV_OUT, OUT_GIF, OUT_MP4)

if __name__ == "__main__":
    main()
