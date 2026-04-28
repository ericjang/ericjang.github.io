# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy", "pillow"]
# ///
"""
Animated comparison of two exploration-bonus terms:

  UCB1 :  sqrt( 2 * ln(t) / n_a )
  PUCT :  sqrt(N) / (1 + n_a)             (the structural part of u(s,a),
                                           ignoring the c_puct * P(s,a) prefactor)

Both are plotted as a function of n_a (visits to action a).  Each frame
fixes the total visit count t == N and sweeps n_a in [1, t].  The animation
walks t from 1 -> T_MAX so you can watch how the two bonus shapes shrink
at different rates.

Outputs
-------
exploration_bonus.json  per-frame curve data (so the JSX side can re-use it)
exploration_bonus.gif   the rendered animation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


HERE = Path(__file__).parent
T_MAX = 80           # upper bound on total visits
FRAMES = T_MAX       # one frame per integer t
FPS = 12


def ucb1_bonus(t: int, n_a: np.ndarray) -> np.ndarray:
    """Classic UCB1 exploration term."""
    # ln(t) is undefined at t=0 and zero at t=1 -> bonus is exactly 0.
    if t <= 1:
        return np.zeros_like(n_a, dtype=float)
    return np.sqrt(2.0 * np.log(t) / n_a)


def puct_bonus(N: int, n_a: np.ndarray) -> np.ndarray:
    """Structural part of the PUCT exploration term."""
    return np.sqrt(N) / (1.0 + n_a)


def compute_frames() -> list[dict]:
    """Compute per-frame curves for both formulas."""
    frames = []
    for t in range(1, T_MAX + 1):
        n_a = np.arange(1, t + 1, dtype=float)
        frames.append({
            "t": t,
            "n_a": n_a.tolist(),
            "ucb1": ucb1_bonus(t, n_a).tolist(),
            "puct": puct_bonus(t, n_a).tolist(),
        })
    return frames


def render_animation(frames: list[dict], out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=140)
    fig.patch.set_facecolor("#f6f1e7")
    ax.set_facecolor("#f6f1e7")

    # Pre-compute global y range so the axes don't jitter.
    all_y = []
    for f in frames:
        all_y.extend(f["ucb1"])
        all_y.extend(f["puct"])
    y_max = max(all_y) * 1.05

    ax.set_xlim(1, T_MAX)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("$n_a$  (visits to action $a$)", fontsize=11)
    ax.set_ylabel("exploration bonus", fontsize=11)

    UCB_COLOR = "#34557a"
    PUCT_COLOR = "#b04a2a"

    (ucb_line,) = ax.plot([], [], color=UCB_COLOR, linewidth=2.0,
                          label=r"UCB1:  $\sqrt{2\,\ln t \,/\, n_a}$")
    (puct_line,) = ax.plot([], [], color=PUCT_COLOR, linewidth=2.0,
                           label=r"PUCT:  $\sqrt{N}\,/\,(1 + n_a)$")
    ucb_dot = ax.scatter([], [], color=UCB_COLOR, s=28, zorder=5)
    puct_dot = ax.scatter([], [], color=PUCT_COLOR, s=28, zorder=5)

    counter = ax.text(
        0.985, 0.50, "", transform=ax.transAxes,
        ha="right", va="center",
        fontsize=12, family="monospace", color="#1f1a14",
        bbox=dict(facecolor="#f6f1e7", edgecolor="#1f1a14",
                  boxstyle="round,pad=0.4", linewidth=0.8),
    )

    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.grid(True, alpha=0.18)
    ax.set_title("Exploration bonus vs visits — UCB1 vs PUCT",
                 fontsize=13, pad=12, color="#1f1a14")

    def update(i):
        f = frames[i]
        n_a = f["n_a"]
        ucb = f["ucb1"]
        puct = f["puct"]
        ucb_line.set_data(n_a, ucb)
        puct_line.set_data(n_a, puct)
        # Highlight the rightmost point (i.e., n_a = t) so you can see
        # the bonus an "always picked" arm currently has.
        if n_a:
            ucb_dot.set_offsets([[n_a[-1], ucb[-1]]])
            puct_dot.set_offsets([[n_a[-1], puct[-1]]])
        counter.set_text(f"t = N = {f['t']:>3}")
        return ucb_line, puct_line, ucb_dot, puct_dot, counter

    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / FPS, blit=False,
    )
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)


def main() -> None:
    frames = compute_frames()

    json_path = HERE / "exploration_bonus.json"
    json_path.write_text(json.dumps(frames, indent=None))
    print(f"wrote {json_path}  ({len(frames)} frames)")

    gif_path = HERE / "exploration_bonus.gif"
    render_animation(frames, gif_path)
    print(f"wrote {gif_path}")


if __name__ == "__main__":
    main()
