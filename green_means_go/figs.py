"""Figures for 'Green means go' from saved aggregate outputs (no model runs). Writes fig1-3 PDFs next to this file."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); R = "/Users/user/Elysium/CV_Research/athlete_ecg"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
INK, MUTED, A, B, RED = "#222222", "#888888", "#3b6ea8", "#e0a030", "#c0392b"
f1b = json.load(open(f"{R}/results/f1b.json"))

# ---- Figure 1: over-call by cohort at the 10%-control operating point
rows = [("PTB-XL\ngreen-only\nn = 1,802", 0.337, 0.315, 0.358), ("Chapman/\nNingbo\ngreen-only\nn = 2,000", 0.179, 0.163, 0.197),
        ("Norwegian\nendurance\nnormal\nn = 26", 0.346, 0.194, 0.538),
        ("PF12RED\nfootballers\ngreen-only\n156 rec. / 52", 0.474, 0.364, 0.585)]
fig, ax = plt.subplots(figsize=(3.6, 2.9))
for i, (lab, p, lo, hi) in enumerate(rows):
    ax.bar(i, p * 100, color=A if i < 2 else B, width=0.62)
    ax.errorbar(i, p * 100, yerr=[[p * 100 - lo * 100], [hi * 100 - p * 100]], color=INK, capsize=3, lw=1)
    ax.text(i, hi * 100 + 2, f"{p * 100:.1f}%", ha="center", color=INK)
ax.axhline(10, color=MUTED, ls="--", lw=1); ax.text(1.5, 11.2, "controls flagged (10%)", ha="center", color=MUTED, fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_ylabel("called abnormal (%)"); ax.set_ylim(0, 70)
ax.text(0.5, 66, "non-athletes", ha="center", color=A, fontsize=7); ax.text(2.5, 66, "athletes", ha="center", color=B, fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig1_overcall.pdf"); plt.close(fig)

# ---- Figure 2: what the model assigns to normal athletes
foot = [(k.title(), v) for k, v in f1b["top_labels_flagged_primary"]
        if k.upper() not in ("SINUS RHYTHM", "NORMAL SINUS RHYTHM", "OTHERWISE NORMAL ECG")][:7]
endu = [("Sinus Bradycardia", 7), ("Marked Sinus Bradycardia", 6), ("Right Axis Deviation", 2), ("Left Atrial Enlargement", 2), ("Anterior Infarct", 1)]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, data, title in ((axs[0], endu, "Norwegian endurance athletes (cardiologist-normal; n = 26)"),
                        (axs[1], foot, "PF12RED footballers (flagged green-only rec.)")):
    labs = [d[0] for d in data][::-1]; vals = [d[1] for d in data][::-1]
    cols = [RED if "Infarct" in l else MUTED for l in labs]
    ax.barh(range(len(vals)), vals, color=cols, height=0.65)
    for j, v in enumerate(vals): ax.text(v + 0.3, j, str(v), va="center", fontsize=7)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7); ax.set_title(title, fontsize=7.5, loc="left")
    ax.set_xlabel("count")
fig.tight_layout(); fig.savefig(f"{HERE}/fig2_labels.pdf"); plt.close(fig)

# ---- Figure 3: operating curves anchored on screening-relevant red ECGs (step 3) + the confirmatory points (step 3b)
f3 = json.load(open(f"{R}/results/f3.json")); f3b = json.load(open(f"{R}/results/f3b.json"))
fig, ax = plt.subplots(figsize=(4.6, 3.3))
style = {"M0_model": ("model's own abnormal output", A, "o", "-"), "Fix2_gate100": ("criteria rule over all findings", RED, "s", "--"),
         "M1_screen_gate": ("criteria rule, screening findings only", "#9b59b6", "^", "--"),
         "M2_measure_then_ask": ("measured intervals + morphology", "#7f8c8d", "D", ":"),
         "Fix1_linear": ("linear readout, criteria labels", "#16a085", "v", "-"), "M3_mlp": ("MLP readout, criteria labels", B, "P", "-")}
for k, (lab, col, mk, ls) in style.items():
    pts = sorted((v["sens_ptb_test"], v["afa_pooled"][0]) for v in f3["methods"][k].values())
    ax.plot([100 * a for a, _ in pts], [100 * b for _, b in pts], ls=ls, marker=mk, ms=3.5, color=col, lw=1.1, label=lab)
for k, col, mk in (("M0_model", A, "o"), ("M3_mlp_s0", B, "*")):
    r = f3b["methods"][k]["0.9"]; a, (lo, hi) = r["afa_pooled"][0], r["afa_pooled"][1]
    ax.errorbar(100 * r["sens_ptb_test"], 100 * a, yerr=[[100 * (a - lo)], [100 * (hi - a)]], fmt=mk, ms=9 if mk == "*" else 6,
                color=col, mec=INK, mew=0.8, capsize=3, lw=1, zorder=5)
ax.annotate("confirmatory test\n(held-out thresholds)", xy=(100 * f3b["methods"]["M3_mlp_s0"]["0.9"]["sens_ptb_test"], 100 * f3b["methods"]["M3_mlp_s0"]["0.9"]["afa_pooled"][0]),
            xytext=(76, 30), fontsize=6.5, color=INK, arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
ax.set_xlabel("sensitivity for screening-relevant abnormal ECGs, PTB-XL test folds (%)"); ax.set_ylabel("normal athletes called abnormal (%)")
ax.set_xlim(70, 96); ax.set_ylim(0, 80); ax.legend(frameon=False, fontsize=6, loc="upper left")
fig.tight_layout(); fig.savefig(f"{HERE}/fig3_fixes.pdf"); plt.close(fig)
print("figures written")
