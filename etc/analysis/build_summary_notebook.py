"""Builds `12_long_range_summary.ipynb` from the cached `results/*.json`
files listed in `model_registry.py`. Run with the standard torch env
(`/home/boittier/metawork/.venv/bin/python`) from this directory:

    python build_summary_notebook.py

Re-run this whenever a new model/notebook is added to the series (most
recently: `PORTED_TORCH_MODELS`, notebook 15) so the summary stays in sync.
Plain nbformat-4 JSON, built by hand (no `nbformat`/`jupyter` package
installed on this machine) -- same approach as every other notebook in this
series.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from model_registry import MODELS, LOREM_JAX_MODELS, PORTED_TORCH_MODELS, RESULTS

ALL_MODELS = MODELS + LOREM_JAX_MODELS + PORTED_TORCH_MODELS

OUTDIR = Path(__file__).parent
FIGDIR = OUTDIR / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

KERNELSPEC = {
    "display_name": "Python 3 (metawork .venv)",
    "language": "python",
    "name": "python3",
}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src, outputs=None):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": outputs or [], "source": src.splitlines(keepends=True)}


def stream_out(text):
    return {"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)}


def image_out(png_path):
    import base64

    data = base64.b64encode(Path(png_path).read_bytes()).decode("ascii")
    return {"output_type": "display_data",
            "data": {"image/png": data, "text/plain": ["<Figure>"]}, "metadata": {}}


plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})

FAMILY_COLORS = {"BPNN": "#0072B2", "PET": "#009E73", "LOREM": "#D55E00", "LOREM-jax": "#9467BD"}

results = [json.loads((RESULTS / f"{m['key']}.json").read_text()) for m in ALL_MODELS]

# --- table text (real, computed here) ---
table_lines = [
    f"{'model':32s} {'family':6s} {'dipole':7s} {'cutoff (Å)':11s} "
    f"{'plateau |F| (eV/Å)':19s} {'ΔE bonded->sep (eV)':20s}"
]
for r in results:
    cutoff = r.get("cutoff_distance")
    cutoff_s = f"{cutoff:.1f}" if cutoff is not None else "n/a"
    plateau = r.get("force_plateau")
    de = r["energy_bonded"] - r["energy_separated"] if "energy_bonded" in r else None
    table_lines.append(
        f"{r['key']:32s} {r['family']:6s} {str(r['dipole']):7s} {cutoff_s:11s} "
        f"{plateau:19.4f} {de:20.4f}"
    )
table_text = "\n".join(table_lines) + "\n"

# --- Fig A: cutoff bar chart ---
fig, ax = plt.subplots(figsize=(9, 4.6))
labels = [r["title"] for r in results]
cutoffs = [r.get("cutoff_distance") or 0 for r in results]
colors = [FAMILY_COLORS.get(r["family"], "0.5") for r in results]
bars = ax.bar(range(len(results)), cutoffs, color=colors)
for bar, r in zip(bars, results):
    if r["dipole"]:
        bar.set_hatch("//")
        bar.set_edgecolor("white")
ax.set_xticks(range(len(results)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax.set_ylabel("empirical cutoff (Å)")
ax.set_title("Empirical long-range cutoff by model")
legend_elems = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]
legend_elems.append(Patch(facecolor="0.6", hatch="//", edgecolor="white", label="has dipole term"))
ax.legend(handles=legend_elems, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIGDIR / "summary_figA_cutoff_bar.png", bbox_inches="tight")
plt.close(fig)

# --- Fig B: force decay overlay ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for r in results:
    style = "--" if r["dipole"] else "-"
    ax.semilogy(r["distance"], r["max_delta_force"],
                color=FAMILY_COLORS.get(r["family"], "0.5"), linestyle=style,
                linewidth=1.8, label=r["title"], alpha=0.85)
ax.set_xlabel("C···I distance (Å)")
ax.set_ylabel("max |ΔF| on CH₃F, vs. fully separated (eV/Å)")
ax.set_title("Force response decay -- all models overlaid")
ax.legend(fontsize=7, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / "summary_figB_force_decay.png", bbox_inches="tight")
plt.close(fig)

# --- Fig C: energy overlay ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for r in results:
    style = "--" if r["dipole"] else "-"
    ax.plot(r["distance"], r["energy"],
            color=FAMILY_COLORS.get(r["family"], "0.5"), linestyle=style,
            linewidth=1.8, label=r["title"], alpha=0.85)
ax.set_xlabel("C···I distance (Å)")
ax.set_ylabel("total energy (eV)")
ax.set_title("Energy vs. leaving-group distance -- all models overlaid")
ax.legend(fontsize=7, loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / "summary_figC_energy.png", bbox_inches="tight")
plt.close(fig)

# --- Fig D: interaction energy overlay ---
fig, ax = plt.subplots(figsize=(8, 5.2))
for r in results:
    style = "--" if r["dipole"] else "-"
    ax.plot(r["distance"], r["interaction_energy"],
            color=FAMILY_COLORS.get(r["family"], "0.5"), linestyle=style,
            linewidth=1.8, label=r["title"], alpha=0.85)
ax.axhline(0, color="0.6", linewidth=1, linestyle=":")
ax.set_xlabel("C···I distance (Å)")
ax.set_ylabel("interaction energy (eV)")
ax.set_title("Size-consistency: interaction energy -- all models overlaid")
ax.legend(fontsize=7, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / "summary_figD_interaction.png", bbox_inches="tight")
plt.close(fig)

# --- notebook cells ---
cells = []

cells.append(md(
    "# Long-range dependence: cross-model summary\n\n"
    "A side-by-side comparison of the reaction-coordinate scan "
    "(`long_range_tests.py`, same protocol as `02_long_range_scan.ipynb`) "
    "across every SN2 model with a usable `model.pt` under `../data/`:\n\n"
    "| notebook | model |\n"
    "|---|---|\n"
    "| `02_long_range_scan.ipynb` | sn2-matched-bpnn |\n"
    "| `03_long_range_scan_sn2.ipynb` | sn2 |\n"
    "| `04_long_range_scan_bpnn_dipole.ipynb` | sn2-matched-bpnn-dipole |\n"
    "| `05_long_range_scan_bpnn_dipole_v2.ipynb` | sn2-matched-bpnn-dipole-v2 |\n"
    "| `06_long_range_scan_bpnn_dipole_v3.ipynb` | sn2-matched-bpnn-dipole-v3 |\n"
    "| `07_long_range_scan_lorem.ipynb` | sn2-matched-lorem |\n"
    "| `08_long_range_scan_lorem_dipole.ipynb` | sn2-matched-lorem-dipole |\n"
    "| `09_long_range_scan_pet.ipynb` | sn2-matched-pet |\n"
    "| `10_long_range_scan_pet_dipole.ipynb` | sn2-matched-pet-dipole |\n"
    "| `11_long_range_scan_lorem_eqmp_smoketest.ipynb` | lorem-eqmp-smoketest |\n"
    "| `13_long_range_scan_lorem_jax_lr.ipynb` | lorem-jax, long-range (Ewald) |\n"
    "| `14_long_range_scan_lorem_jax_sr.ipynb` | lorem-jax, short-range only |\n"
    "| `15_long_range_scan_lorem_jax_ported_torch.ipynb` | lorem-jax R2_E+F checkpoint, ported to torch |\n\n"
    "Directories with a training run but no exported `model.pt` "
    "(`sn2-bpnn`, `sn2-bpnn-dipole`, `sn2-dipole`, `sn2-pet`, "
    "`sn2-pet-dipole`, `diag-bpnn-w20`) have nothing to load and are not "
    "included.\n\n"
    "**Notebooks `13`/`14` are a different codebase, not the same "
    "training run as `sn2-matched-lorem`.** They load the official "
    "[`lorem-tmlr-archive`](https://github.com/sirmarcel/lorem-tmlr-archive) "
    "reproduction checkpoints (native `lorem-jax`, not metatrain/torch) "
    "for a broader SN2 halide-exchange benchmark, via "
    "`lorem.calculator.Calculator` instead of `MetatomicCalculator`, and "
    "need a different interpreter "
    "(`/home/boittier/metawork/.venv-lorem-jax/bin/python`, not the "
    "`metawork/.venv` used by every other notebook here). Their absolute "
    "energies are not comparable to the `sn2-matched-*` family (different "
    "training data); only the empirical cutoffs are.\n\n"
    "**Notebook `15` loads the exact same `13` checkpoint a third way**: a "
    "from-scratch torch port (`JaxParityBackbone`/`JaxParityLongRange`, "
    "`metatrain/.../lorem/modules/jax_parity.py`), run in the *same* "
    "`metawork/.venv` torch environment as `02`-`11` (no jax needed at "
    "runtime). It is validated against the live jax reference to ~2e-5 eV / "
    "~3e-5 eV/Å -- see its own notebook for the full validation, including "
    "two non-PBC long-range bugs found and fixed along the way.\n\n"
    "This notebook does not reload any models -- it reads the cached "
    "numeric results each per-model notebook writes to "
    "`results/<model>.json` (distance/energy/force arrays, the empirical "
    "cutoff, etc.), so it stays fast to re-run after any single model is "
    "updated."
))

cells.append(md(
    "## Load cached results\n\n"
    "`model_registry.py` lists every model this series covers, in a fixed "
    "order used consistently for colors/labels below."
))

load_src = (
    "import json\n"
    "from pathlib import Path\n\n"
    "from model_registry import MODELS, LOREM_JAX_MODELS, PORTED_TORCH_MODELS, RESULTS\n\n"
    "ALL_MODELS = MODELS + LOREM_JAX_MODELS + PORTED_TORCH_MODELS\n"
    "results = [json.loads((RESULTS / f\"{m['key']}.json\").read_text()) for m in ALL_MODELS]\n"
    "[r['key'] for r in results]\n"
)
keys_list = json.dumps([r["key"] for r in results])
cells.append(code(load_src, outputs=[{
    "output_type": "execute_result", "execution_count": None, "metadata": {},
    "data": {"text/plain": [keys_list]},
}]))

cells.append(md(
    "## Comparison table\n\n"
    "Empirical cutoff, the residual force plateau past that cutoff (the "
    "CH₃F fragment's own non-equilibrium force, not I⁻ coupling), and the "
    "total energy change from the most compressed to the most separated "
    "geometry sampled (1.5 Å to 32 Å)."
))

table_src = (
    "header = (f\"{'model':32s} {'family':6s} {'dipole':7s} {'cutoff (Å)':11s} \"\n"
    "          f\"{'plateau |F| (eV/Å)':19s} {'ΔE bonded->sep (eV)':20s}\")\n"
    "print(header)\n"
    "for r in results:\n"
    "    cutoff = r.get('cutoff_distance')\n"
    "    cutoff_s = f'{cutoff:.1f}' if cutoff is not None else 'n/a'\n"
    "    plateau = r.get('force_plateau')\n"
    "    de = r['energy_bonded'] - r['energy_separated'] if 'energy_bonded' in r else None\n"
    "    print(f\"{r['key']:32s} {r['family']:6s} {str(r['dipole']):7s} {cutoff_s:11s} \"\n"
    "          f\"{plateau:19.4f} {de:20.4f}\")\n"
)
cells.append(code(table_src, outputs=[stream_out(table_text)]))

cells.append(md(
    "## Empirical cutoff by model\n\n"
    "Bar height is the empirical cutoff from `estimate_cutoff` (threshold "
    "0.01 eV/Å on the I-dependent force change); color is the model "
    "family, hatching marks the dipole/long-range-term variants."
))
figA_src = (
    "import matplotlib.pyplot as plt\n"
    "from matplotlib.patches import Patch\n\n"
    "FAMILY_COLORS = {\"BPNN\": \"#0072B2\", \"PET\": \"#009E73\", \"LOREM\": \"#D55E00\", "
    "\"LOREM-jax\": \"#9467BD\"}\n\n"
    "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
    "labels = [r[\"title\"] for r in results]\n"
    "cutoffs = [r.get(\"cutoff_distance\") or 0 for r in results]\n"
    "colors = [FAMILY_COLORS.get(r[\"family\"], \"0.5\") for r in results]\n"
    "bars = ax.bar(range(len(results)), cutoffs, color=colors)\n"
    "for bar, r in zip(bars, results):\n"
    "    if r[\"dipole\"]:\n"
    "        bar.set_hatch(\"//\")\n"
    "        bar.set_edgecolor(\"white\")\n"
    "ax.set_xticks(range(len(results)))\n"
    "ax.set_xticklabels(labels, rotation=40, ha=\"right\", fontsize=8)\n"
    "ax.set_ylabel(\"empirical cutoff (Å)\")\n"
    "ax.set_title(\"Empirical long-range cutoff by model\")\n"
    "legend_elems = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items()]\n"
    "legend_elems.append(Patch(facecolor=\"0.6\", hatch=\"//\", edgecolor=\"white\", label=\"has dipole term\"))\n"
    "ax.legend(handles=legend_elems, fontsize=8, loc=\"upper left\")\n"
    "fig.tight_layout()\n"
)
cells.append(code(figA_src, outputs=[image_out(FIGDIR / "summary_figA_cutoff_bar.png")]))

cells.append(md(
    "## Force-response decay, all models overlaid\n\n"
    "Solid lines: no dipole term. Dashed lines: dipole/long-range variant. "
    "LOREM's characteristic slower decay (blue/green cluster near ~5.5 Å "
    "vs. the orange/purple LOREM curves reaching past ~20 Å) is the main "
    "qualitative result of this comparison."
))
figB_src = (
    "fig, ax = plt.subplots(figsize=(8, 5.2))\n"
    "for r in results:\n"
    "    style = \"--\" if r[\"dipole\"] else \"-\"\n"
    "    ax.semilogy(r[\"distance\"], r[\"max_delta_force\"],\n"
    "                color=FAMILY_COLORS.get(r[\"family\"], \"0.5\"), linestyle=style,\n"
    "                linewidth=1.8, label=r[\"title\"], alpha=0.85)\n"
    "ax.set_xlabel(\"C···I distance (Å)\")\n"
    "ax.set_ylabel(\"max |ΔF| on CH₃F, vs. fully separated (eV/Å)\")\n"
    "ax.set_title(\"Force response decay -- all models overlaid\")\n"
    "ax.legend(fontsize=7, loc=\"upper right\", ncol=2)\n"
    "fig.tight_layout()\n"
)
cells.append(code(figB_src, outputs=[image_out(FIGDIR / "summary_figB_force_decay.png")]))

cells.append(md("## Energy vs. leaving-group distance, all models overlaid"))
figC_src = (
    "fig, ax = plt.subplots(figsize=(8, 5.2))\n"
    "for r in results:\n"
    "    style = \"--\" if r[\"dipole\"] else \"-\"\n"
    "    ax.plot(r[\"distance\"], r[\"energy\"],\n"
    "            color=FAMILY_COLORS.get(r[\"family\"], \"0.5\"), linestyle=style,\n"
    "            linewidth=1.8, label=r[\"title\"], alpha=0.85)\n"
    "ax.set_xlabel(\"C···I distance (Å)\")\n"
    "ax.set_ylabel(\"total energy (eV)\")\n"
    "ax.set_title(\"Energy vs. leaving-group distance -- all models overlaid\")\n"
    "ax.legend(fontsize=7, loc=\"lower right\", ncol=2)\n"
    "fig.tight_layout()\n"
)
cells.append(code(figC_src, outputs=[image_out(FIGDIR / "summary_figC_energy.png")]))

cells.append(md(
    "## Size-consistency: interaction energy, all models overlaid\n\n"
    "`fragment_separation_test`'s interaction energy for every model, "
    "which should decay to ~0 past each model's own cutoff (or stay at a "
    "genuine long-range tail, for LOREM)."
))
figD_src = (
    "fig, ax = plt.subplots(figsize=(8, 5.2))\n"
    "for r in results:\n"
    "    style = \"--\" if r[\"dipole\"] else \"-\"\n"
    "    ax.plot(r[\"distance\"], r[\"interaction_energy\"],\n"
    "            color=FAMILY_COLORS.get(r[\"family\"], \"0.5\"), linestyle=style,\n"
    "            linewidth=1.8, label=r[\"title\"], alpha=0.85)\n"
    "ax.axhline(0, color=\"0.6\", linewidth=1, linestyle=\":\")\n"
    "ax.set_xlabel(\"C···I distance (Å)\")\n"
    "ax.set_ylabel(\"interaction energy (eV)\")\n"
    "ax.set_title(\"Size-consistency: interaction energy -- all models overlaid\")\n"
    "ax.legend(fontsize=7, loc=\"upper right\", ncol=2)\n"
    "fig.tight_layout()\n"
)
cells.append(code(figD_src, outputs=[image_out(FIGDIR / "summary_figD_interaction.png")]))

bpnn_pet_cutoffs = [r["cutoff_distance"] for r in results if r["family"] in ("BPNN", "PET")]
lorem_cutoffs = [r["cutoff_distance"] for r in results if r["family"] == "LOREM"]
jax_lr = next(r for r in results if r["key"] == "lorem-jax-sn2-lr")
jax_sr = next(r for r in results if r["key"] == "lorem-jax-sn2-sr")
jax_ported = next(r for r in results if r["key"] == "lorem-jax-sn2-lr-ported-torch")

cells.append(md(
    "## Takeaways\n\n"
    f"- **BPNN and PET models are short-range**, with an empirical cutoff "
    f"clustered tightly at **{min(bpnn_pet_cutoffs):.1f}-{max(bpnn_pet_cutoffs):.1f} Å** "
    "across all 7 BPNN/PET variants tested (matched, dipole-augmented, and "
    "the original unmatched `sn2` model alike) -- consistent with a "
    "symmetry-function / message-passing receptive field, whether or not "
    "a dipole/electrostatic head was added on top.\n"
    f"- **LOREM models are genuinely long-range**: both `sn2-matched-lorem` "
    "variants and the `lorem-eqmp-smoketest` model (all trained with "
    "metatrain/torch) show an empirical "
    f"cutoff of **{min(lorem_cutoffs):.1f}-{max(lorem_cutoffs):.1f} Å** -- roughly "
    "4x further than the BPNN/PET cluster -- confirming LOREM's "
    "long-range electrostatic term is doing real work rather than just "
    "being numerically slow to vanish.\n"
    f"- **The official native `lorem-jax` checkpoints independently confirm "
    "the same pattern**, from a completely different codebase, training "
    f"run, and dataset: the long-range-enabled checkpoint gives a "
    f"**{jax_lr['cutoff_distance']:.1f} Å** cutoff (even further than the "
    "metatrain-trained LOREM models -- this is the paper's own Ewald "
    "long-range term, unrestricted by a training-domain cutoff), while "
    "its short-range-only sibling checkpoint (`lr: false`, no Ewald path) "
    f"lands right back in the BPNN/PET cluster at "
    f"**{jax_sr['cutoff_distance']:.1f} Å**. This is strong, "
    "architecture-independent evidence that the long-vs-short cutoff "
    "split tracks whether a genuine long-range electrostatic term is "
    "present, not an artifact of any one codebase or training run.\n"
    "- **A torch port of the real lorem-jax checkpoint now exists and is "
    "validated** (notebook `15`, `lorem_jax_ported_torch.py`): loading the "
    "exact same `lorem-jax-sn2-lr` checkpoint's weights into "
    "`JaxParityBackbone`/`JaxParityLongRange` "
    "(`metatrain/.../lorem/modules/jax_parity.py`) and running it in the "
    "same `metawork/.venv` torch environment as every BPNN/PET/LOREM model "
    f"above gives an empirical cutoff of **{jax_ported['cutoff_distance']:.1f} Å** "
    "-- matching `lorem-jax-sn2-lr` to within 0.1 Å, as expected since it "
    "is a numerically-validated (~2e-5 eV / ~3e-5 eV/Å) port of the exact "
    "same trained model, not a retraining. This supersedes the earlier "
    "note that no such loader existed (see notebook `15`'s markdown for "
    "the full validation, including two non-PBC long-range bugs found and "
    "fixed along the way: a wrongly-inherited near-field exclusion radius, "
    "and a missing x2 training/production convention factor).\n"
    "- Adding a dipole term to BPNN or PET does **not** measurably extend "
    "the cutoff (all still ≈5.5 Å); it changes the energetics "
    "(`ΔE bonded->separated` shifts by up to ~1.5 eV between variants) "
    "without adding genuine long-range force dependence -- i.e. the "
    "dipole head here is refining short-range energetics/dipole "
    "prediction, not introducing long-range coupling.\n"
    "- All models remain **size-consistent** in the `fragment_separation_test` "
    "sense: interaction energy decays toward 0 at large separation (LOREM "
    "over its longer range, BPNN/PET over the shorter one) rather than "
    "diverging or plateauing at a nonzero value.\n"
    "- Per-model details, including the reaction-coordinate energy/force "
    "curves and structure snapshots, are in notebooks "
    "`02_long_range_scan.ipynb`, `03`-`11`, and `13`-`15` listed above."
))

cells.append(code(""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": KERNELSPEC,
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = OUTDIR / "12_long_range_summary.ipynb"
with open(out_path, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out_path)
