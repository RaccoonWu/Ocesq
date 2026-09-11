"""
Generate Fig 4 (materialization) and Fig 5 (downstream recovery).

Palette: CCFA standard (palette-and-accessibility.md)
  #1F6F8B  blue   — proposed method (APC / Complete package)
  #D2673D  orange — primary baseline (Full OEG)
  #6657A8  purple — OC-RES
  #2A9D8F  teal   — Two-hop
  #B58B2A  gold   — Change metric
  #BA4C5E  rose   — (reserved)
  #477AA6  steel  — Action-window
  #5D6977  gray   — weak baselines (Raw trace / Raw diff)
  #8B96A3  gray   — secondary
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#5D6977",
    "xtick.color": "#5D6977",
    "ytick.color": "#5D6977",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
})

# ---- CCFA palette ----
C = {
    "apc":     "#1F6F8B",  # proposed method — stable across all figures
    "oeg":     "#D2673D",  # primary baseline
    "purple":  "#6657A8",  # OC-RES
    "teal":    "#2A9D8F",  # Two-hop / Full answer
    "gold":    "#B58B2A",  # Change metric
    "rose":    "#BA4C5E",
    "steel":   "#477AA6",  # Action-window
    "grey":    "#5D6977",  # weak baselines (Raw trace / diff)
    "lgrey":   "#8B96A3",  # secondary elements
    # Neutral support
    "ink":     "#222222",
    "grid":    "#D9D9D9",
    "border":  "#C9D1D9",
}

OUT = "figures"


def save(fig, stem):
    fig.savefig(f"{OUT}/{stem}.pdf")
    print(f"  -> {stem}.pdf")


# ===================================================================
# FIGURE 4 — Materialization: preservation + graph size
# ===================================================================
def make_fig4():
    methods = [
        "Two-hop\nneighb.",
        "Rooted-\nSteiner",
        "Why-not\nclosure",
        "OC-RES\nslice",
        "Complete\npackage",
        "Full\ngraph",
    ]
    x = np.arange(len(methods))
    bar_w = 0.16

    # ---- data from Table 3 ----
    decision = [0.027, 0.000, 0.189, 0.820, 1.000, 1.000]
    full     = [0.000, 0.000, 0.180, 0.000, 1.000, 1.000]
    change   = [0.495, 0.207, 0.838, 0.468, 1.000, 1.000]
    conflict = [1.000, 0.712, 1.000, 0.829, 1.000, 1.000]
    nodes    = [27.45,  9.04, 50.68,  6.57, 10.96,  73.00]
    edges    = [36.17,  8.04, 81.46,  4.94,  1.87, 110.22]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.15))

    # ---- (a) Preservation rates ----
    offsets = [-1.5, -0.5, 0.5, 1.5]
    metrics = [decision, full, change, conflict]
    labels  = ["Decision", "Full answer", "Change", "Conflict"]
    colors  = [C["apc"], C["teal"], C["gold"], C["purple"]]  # 4 distinct CCFA hues

    for off, vals, lbl, col in zip(offsets, metrics, labels, colors):
        bx = x + off * bar_w
        ax1.bar(bx, vals, bar_w, label=lbl, color=col, edgecolor="white", linewidth=0.15)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=5.5, color=C["ink"])
    ax1.set_ylabel("Preservation rate", fontsize=7, color=C["ink"])
    ax1.set_ylim(0, 1.12)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
    ax1.legend(ncols=4, frameon=True, fancybox=False, edgecolor=C["border"],
               loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.18),
               fontsize=5, handlelength=0.9, columnspacing=0.5,
               borderaxespad=0, mode="expand")
    ax1.set_title("(a) Answer preservation by method", fontweight="bold",
                  loc="left", fontsize=7.5, pad=3, color=C["ink"])
    ax1.grid(axis="y", alpha=0.2, linewidth=0.3, color=C["grid"])
    ax1.set_xlim(-0.55, len(methods) - 0.45)
    ax1.tick_params(axis="both", pad=1.5, colors=C["grey"])

    # ---- (b) Graph size ----
    bw2 = 0.30
    bx_n = x - bw2 / 2
    bx_e = x + bw2 / 2

    ax2.bar(bx_n, nodes, bw2, label="Nodes", color=C["apc"],
            edgecolor="white", linewidth=0.15)
    ax2.bar(bx_e, edges, bw2, label="Edges", color=C["oeg"],
            edgecolor="white", linewidth=0.15)

    for bx_vals, vals in [(bx_n, nodes), (bx_e, edges)]:
        for xi, vi in zip(bx_vals, vals):
            va = "bottom" if vi < 50 else "top"
            yoff = 0.5 if vi < 50 else -0.5
            ax2.text(xi + bw2 / 2, vi + yoff, f"{vi:.1f}",
                     ha="center", fontsize=4.2, fontweight="bold",
                     color=C["ink"], va=va)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=5.5, color=C["ink"])
    ax2.set_ylabel("Count", fontsize=7, color=C["ink"])
    ax2.legend(frameon=True, fancybox=False, edgecolor=C["border"], fontsize=5.5,
               handlelength=0.9)
    ax2.set_title("(b) Graph size (avg. nodes / edges)", fontweight="bold",
                  loc="left", fontsize=7.5, pad=3, color=C["ink"])
    ax2.grid(axis="y", alpha=0.2, linewidth=0.3, color=C["grid"])
    ax2.set_xlim(-0.55, len(methods) - 0.45)
    ax2.tick_params(axis="both", pad=1.5, colors=C["grey"])

    fig.tight_layout(pad=0.8, w_pad=1.8, rect=(0.005, 0.01, 0.995, 0.90))
    save(fig, "fig4_materialization")
    plt.close(fig)


# ===================================================================
# FIGURE 5 — Downstream fact & source recovery
# ===================================================================
def make_fig5():
    # --- Pilot data: all 6 baselines, source_f1 (consistent methodology) ---
    # Ordered by APC→OC-RES→Two-hop→Action-window→Full OEG→Raw trace
    methods_6 = [
        "Complete\npackage",   # proposed
        "OC-RES\nslice",
        "Two-hop\nneighb.",
        "Action\nwindow",
        "Full\nOEG",           # primary baseline
        "Raw\ntrace",          # weak baseline
    ]
    ds_pilot = [0.840, 0.811, 0.575, 0.286, 0.462, 0.308]
    qw_pilot = [0.784, 0.529, 0.381, 0.422, 0.277, 0.448]

    # --- Formal data: ablation (source_f1_on_faults) ---
    ds_full = 0.859;  ds_nost = 0.860;  ds_nosp = 0.013
    qw_full = 0.766;  qw_nost = 0.757;  qw_nosp = 0.0
    ds_oeg  = 0.554;  qw_oeg  = 0.472

    # --- Formal data: token & FP ---
    token_ds = [38062, 15741]
    token_qw = [38970, 15679]
    fp_ds    = [24, 9]

    # ---------- layout ----------
    fig = plt.figure(figsize=(6.0, 3.8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1.0], wspace=0.30,
                  left=0.06, right=0.98, top=0.92, bottom=0.12)

    # ---- (a) Comprehensive Source F1 ----
    # x-axis = evidence configuration, 2 grouped bars per config (DeepSeek / Qwen)
    ax_a = fig.add_subplot(gs[0, 0])
    x6 = np.arange(len(methods_6))
    bw = 0.28

    # DeepSeek = CCFA blue (consistent model color), Qwen = CCFA orange
    bars_ds = ax_a.bar(x6 - bw/2, ds_pilot, bw, label="DeepSeek v4-flash",
                       color=C["apc"], edgecolor="white", linewidth=0.15)
    bars_qw = ax_a.bar(x6 + bw/2, qw_pilot, bw, label="Qwen 3.7-plus",
                       color=C["oeg"], edgecolor="white", linewidth=0.15)

    for bars in [bars_ds, bars_qw]:
        for b in bars:
            ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                      f"{b.get_height():.3f}", ha="center", fontsize=4.5,
                      fontweight="bold", color=C["ink"])

    ax_a.set_xticks(x6)
    ax_a.set_xticklabels(methods_6, fontsize=5.5, color=C["ink"])
    ax_a.set_ylabel("Source F1", fontsize=7, color=C["ink"])
    ax_a.set_ylim(0, 0.96)
    ax_a.legend(ncols=1, frameon=True, fancybox=False, edgecolor=C["border"],
                fontsize=5.5, handlelength=0.9, loc="upper right")
    ax_a.set_title("(a) Source F1 by evidence configuration", fontweight="bold",
                   loc="left", fontsize=7.5, pad=3, color=C["ink"])
    ax_a.grid(axis="y", alpha=0.2, linewidth=0.3, color=C["grid"])
    ax_a.set_xlim(-0.55, len(methods_6) - 0.45)
    ax_a.tick_params(axis="both", pad=1.5, colors=C["grey"])

    # ---- (b) right panel: ablation table + token/FP ----
    # (b1) Ablation table
    ax_tbl = fig.add_axes([0.57, 0.48, 0.41, 0.41])
    ax_tbl.axis("off")
    ax_tbl.set_xlim(0, 1)
    ax_tbl.set_ylim(0, 1)

    col_x = [0.02, 0.42, 0.68, 0.94]
    rows_y = [0.82, 0.62, 0.44, 0.26, 0.08]

    def draw_cell(r, c, text, bold=False, color=None):
        x = (col_x[c] + col_x[c+1]) / 2
        y = rows_y[r]
        ax_tbl.text(x, y, text, ha="center", va="center",
                    fontsize=6.5, fontweight="bold" if bold else "normal",
                    color=color or C["ink"])

    draw_cell(0, 0, "Condition", bold=True)
    draw_cell(0, 1, "DeepSeek", bold=True)
    draw_cell(0, 2, "Qwen", bold=True)

    row_data = [
        ("APC (full)",     f"{ds_full:.3f}", f"{qw_full:.3f}"),
        ("–Status",        f"{ds_nost:.3f}", f"{qw_nost:.3f}"),
        ("–Source ptr",    f"{ds_nosp:.3f}", "~0"),
        ("Full OEG",       f"{ds_oeg:.3f}",  f"{qw_oeg:.3f}"),
    ]
    for r_idx, (lbl, v1, v2) in enumerate(row_data):
        draw_cell(r_idx + 1, 0, lbl)
        draw_cell(r_idx + 1, 1, v1)
        draw_cell(r_idx + 1, 2, v2)

    # Booktabs horizontal rules
    for ypos, lw in [(0.72, 0.8), (0.53, 0.3), (0.35, 0.3), (0.17, 0.3), (0.02, 0.8)]:
        ax_tbl.axhline(y=ypos, xmin=col_x[0], xmax=col_x[3] + 0.03,
                       color=C["lgrey"], linewidth=lw, clip_on=False)

    ax_tbl.set_title("(b) Ablation & baselines (fault-only F1)", fontweight="bold",
                     loc="left", fontsize=7.5, pad=6, color=C["ink"])

    # (b2) Token + FP mini charts
    gs_bot = GridSpec(1, 2, figure=fig,
                      left=0.57, right=0.98, top=0.32, bottom=0.12, wspace=0.40)

    # Token usage — Full OEG vs APC
    ax_tok = fig.add_subplot(gs_bot[0, 0])
    tok_mean = [(token_ds[0] + token_qw[0]) / 2, (token_ds[1] + token_qw[1]) / 2]
    bars_t = ax_tok.bar([0, 1], tok_mean, 0.50,
                        color=[C["oeg"], C["apc"]],   # orange=baseline, blue=proposed
                        edgecolor="white", linewidth=0.15)
    ax_tok.set_xticks([0, 1])
    ax_tok.set_xticklabels(["Full OEG", "APC"], fontsize=6, color=C["ink"])
    ax_tok.set_ylabel("Mean input\ntokens", fontsize=6, color=C["ink"])
    ax_tok.set_title("Token usage", fontweight="bold", loc="left", fontsize=7, color=C["ink"])
    ax_tok.grid(axis="y", alpha=0.2, linewidth=0.3, color=C["grid"])
    ax_tok.tick_params(axis="y", pad=1.5, colors=C["grey"])
    for b, v in zip(bars_t, tok_mean):
        ax_tok.text(b.get_x() + b.get_width() / 2, b.get_height() + 200,
                    f"{v:.0f}", ha="center", fontsize=6, fontweight="bold",
                    color=C["ink"])

    # Benign FP (DeepSeek only — Qwen APC has 0)
    ax_fp = fig.add_subplot(gs_bot[0, 1])
    bars_f = ax_fp.bar([0, 1], fp_ds, 0.50,
                       color=[C["oeg"], C["apc"]],
                       edgecolor="white", linewidth=0.15)
    ax_fp.set_xticks([0, 1])
    ax_fp.set_xticklabels(["Full OEG", "APC"], fontsize=6, color=C["ink"])
    ax_fp.set_ylabel("Cases", fontsize=6.5, color=C["ink"])
    ax_fp.set_title("Benign false positives\n(DeepSeek)", fontweight="bold",
                    loc="left", fontsize=7, color=C["ink"])
    ax_fp.grid(axis="y", alpha=0.2, linewidth=0.3, color=C["grid"])
    ax_fp.tick_params(axis="y", pad=1.5, colors=C["grey"])
    for b, v in zip(bars_f, fp_ds):
        ax_fp.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                   str(v), ha="center", fontsize=6.5, fontweight="bold",
                   color=C["ink"])

    save(fig, "fig5_downstream_recovery")
    plt.close(fig)


# ===================================================================
if __name__ == "__main__":
    make_fig4()
    make_fig5()
    print("Done — fig4_materialization.pdf + fig5_downstream_recovery.pdf")
