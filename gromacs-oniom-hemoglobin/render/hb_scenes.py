"""POV-Ray scenes of the hemoglobin ONIOM stress test.

hb_scenes.py OUTDIR: the solvated system (cut away), the protein with its four ML heme
sites, one site close up with its link atom, and relaxed heme sites from several models
overlaid edge-on. Reads the compact NpT frame, the SITE groups and models/relaxed_site1/.
"""
import json
import sys
from pathlib import Path

import ase.io
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gromacs-oniom-cyclic" / "render"))
from povlib import P, cyl, header, sph  # noqa: E402

HB = Path("/home/boittier/metawork/gromacs-oniom-hemoglobin")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "hb_scenes")
OUT.mkdir(parents=True, exist_ok=True)
FACTS = {}

lines = (HB / "npt_compact.gro").read_text().splitlines()
n = int(lines[1])
at = lines[2 : 2 + n]
xyz = np.array([[float(l[20 + 8 * k : 28 + 8 * k]) for k in range(3)] for l in at])
name = np.array([l[10:15].strip() for l in at])
resn = np.array([l[5:10].strip() for l in at])
resid = np.array([int(l[:5]) for l in at])
el = np.array(["Fe" if a == "FE" else ("Na" if a == "NA" and r == "NA" else ("Cl" if r == "CL" else a[0])) for a, r in zip(name, resn)])

groups, cur = {}, None
for l in open(HB / "index.ndx"):
    if l.startswith("["):
        cur = l.strip("[] \n")
        groups[cur] = []
    else:
        groups[cur] += [int(t) - 1 for t in l.split()]
sites = [np.array(groups[f"SITE{k}"]) for k in range(1, 5)]
ml = np.concatenate(sites)

protein = np.where(~np.isin(resn, ["SOL", "NA", "CL", "HEM"]))[0]
xyz = xyz - xyz[protein].mean(0)
starts = [0, 2142, 4367, 6509, protein.max() + 1]
chains = [protein[(protein >= a) & (protein < b)] for a, b in zip(starts[:-1], starts[1:])]
CHAIN = ["<0.24,0.50,0.85>", "<0.55,0.36,0.78>", "<0.16,0.62,0.66>", "<0.80,0.40,0.62>"]
CHAIN_NAMES = ["alpha1", "beta1", "alpha2", "beta2"]
SITE_C = "<0.96,0.66,0.12>"
FE_C = "<0.80,0.28,0.10>"
EL = {"C": "<0.50,0.52,0.55>", "N": "<0.20,0.38,0.90>", "O": "<0.90,0.20,0.15>", "H": "<0.93,0.93,0.93>",
      "Fe": FE_C, "S": "<0.90,0.80,0.20>"}
cam = np.array([1.0, -1.3, 0.55])
cd = cam / np.linalg.norm(cam)


def bonds(idx, X=xyz, e=el):
    idx = np.asarray(idx)
    out = []
    P_ = X[idx]
    for a in range(len(idx)):
        d = np.linalg.norm(P_[a + 1 :] - P_[a], axis=1)
        for b in np.where(d < 0.25)[0] + a + 1:
            i, j = idx[a], idx[b]
            pair = {e[i], e[j]}
            lim = 0.24 if "Fe" in pair else (0.125 if "H" in pair else 0.175)
            if "H" in pair and len(pair) == 1:
                continue
            if d[b - a - 1] < lim:
                out.append((i, j))
    return out


def ball_stick(idx, color=None, r=0.035, rs=0.016, X=xyz, e=el, t=0.0, big_fe=0.07):
    s = ""
    for i in idx:
        rad = big_fe if e[i] == "Fe" else (r * 0.6 if e[i] == "H" else r)
        s += sph(X[i], rad, color if (color and e[i] not in ("Fe",)) else EL.get(e[i], EL["C"]), t)
    for i, j in bonds(idx, X, e):
        ci = color if (color and e[i] != "Fe") else EL.get(e[i], EL["C"])
        cj = color if (color and e[j] != "Fe") else EL.get(e[j], EL["C"])
        m = (X[i] + X[j]) / 2
        s += cyl(X[i], m, rs, ci, t) + cyl(m, X[j], rs, cj, t)
    return s


def ca_tubes(r=0.11):
    s = ""
    for c, idx in enumerate(chains):
        ca = idx[name[idx] == "CA"]
        for a, b in zip(ca[:-1], ca[1:]):
            s += cyl(xyz[a], xyz[b], r, CHAIN[c]) + sph(xyz[b], r, CHAIN[c])
        s += sph(xyz[ca[0]], r, CHAIN[c])
    return s


def sites_scene(scale=1.0):
    s = ""
    for idx in sites:
        heavy = idx[el[idx] != "H"]
        s += ball_stick(heavy, SITE_C, r=0.05 * scale, rs=0.025 * scale, big_fe=0.13 * scale)
    return s


# ---------------------------------------------------------------- 1. whole system, cut away
sol_o = np.where(name == "OW")[0]
ions = np.where(np.isin(resn, ["NA", "CL"]))[0]
behind = lambda ids, off=0.0: ids[xyz[ids] @ cd < off]
wat = behind(sol_o, 0.3)
depth = xyz[wat] @ cd
q = (depth - depth.min()) / np.ptp(depth)
txt = header(cam, (0, 0, 0), 10.2, ortho=True, dist=80)
near_c, far_c = np.array([0.33, 0.50, 0.68]), np.array([0.86, 0.91, 0.96])
for b_ in range(8):
    sel = wat[np.minimum((q * 8).astype(int), 7) == b_]
    c = far_c + (near_c - far_c) * (b_ + 0.5) / 8
    col = f"<{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}>"
    txt += "".join(sph(xyz[o], 0.085, col, 0.0, "FW", False) for o in sel)
for i in behind(ions, 0.3):
    txt += sph(xyz[i], 0.15, "<0.55,0.35,0.85>" if resn[i] == "NA" else "<0.30,0.75,0.35>")
txt += ca_tubes(0.10) + sites_scene(1.3)
(OUT / "hb_system.pov").write_text(txt)

# ---------------------------------------------------------------- 2. protein and ML sites
txt = header(cam, (0, 0, 0), 7.6, ortho=True, dist=60) + ca_tubes(0.12) + sites_scene(1.4)
(OUT / "hb_protein.pov").write_text(txt)

# ---------------------------------------------------------------- 3. one site close up
site = sites[0]
fe = site[el[site] == "Fe"][0]
cb = site[name[site] == "CB"][0]
ca = min((j for j in protein if name[j] == "CA" and j not in site), key=lambda j: np.linalg.norm(xyz[j] - xyz[cb]))
link = xyz[cb] + (xyz[ca] - xyz[cb]) / np.linalg.norm(xyz[ca] - xyz[cb]) * 0.1
ringN = site[np.isin(name[site], ["NA", "NB", "NC", "ND"])]
c0 = xyz[ringN].mean(0)
nrm = np.linalg.svd(xyz[ringN] - c0)[2][-1]
ne2 = site[name[site] == "NE2"][0]
if nrm @ (xyz[ne2] - c0) < 0:
    nrm = -nrm
side = np.cross(nrm, xyz[ringN[0]] - c0)
side /= np.linalg.norm(side)
view = side * 0.9 + nrm * 0.45 + np.cross(nrm, side) * 0.35
center = (xyz[fe] + xyz[cb]) / 2
d_site = np.array([np.min(np.linalg.norm(xyz[site] - xyz[i], axis=1)) for i in protein])
env_res = set(zip(resid[protein[d_site < 0.45]], (protein[d_site < 0.45] // 1)))
env = protein[(d_site < 0.6) & ~np.isin(protein, site) & (el[protein] != "H")]
env = env[(xyz[env] - center) @ (view / np.linalg.norm(view)) < 0.25]
txt = header(view, center, 2.3, ortho=True, dist=40, up=tuple(nrm))
txt += ball_stick(env, "<0.62,0.66,0.72>", r=0.022, rs=0.011, t=0.55)
txt += ball_stick(site, None, r=0.04, rs=0.018, big_fe=0.075)
txt += sph(xyz[ca], 0.05, "<0.55,0.58,0.62>", 0.4) + cyl(xyz[cb], xyz[ca], 0.012, "<0.55,0.58,0.62>", 0.4)
txt += sph(link, 0.032, "<0.90,0.15,0.75>") + cyl(xyz[cb], link, 0.016, "<0.90,0.15,0.75>")
(OUT / "hb_site.pov").write_text(txt)
FACTS["site1"] = {"atoms": int(len(site)), "fe_ne2_nm": float(np.linalg.norm(xyz[fe] - xyz[ne2])),
                  "cb_ca_nm": float(np.linalg.norm(xyz[cb] - xyz[ca])), "env_atoms": int(len(env))}

# ---------------------------------------------------------------- 4. relaxed heme sites, edge-on
R = HB / "models" / "relaxed_site1"
if (R / "crystal.xyz").exists():
    ref = ase.io.read(R / "crystal.xyz")
    names = ref.info["names"].split()
    core = [k for k, a in enumerate(names) if a in ("NA", "NB", "NC", "ND") or a.startswith("C1") or a.startswith("C4") or a.startswith("CH")]
    heavy = [k for k, a in enumerate(names) if not a.startswith("H") and a not in ("HL",)]

    def kabsch(Pm, Q):
        Pc, Qc = Pm - Pm.mean(0), Q - Q.mean(0)
        U, S_, Vt = np.linalg.svd(Pc.T @ Qc)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt, Pm.mean(0), Q.mean(0)

    X0 = ref.get_positions() / 10
    kc = [names.index(a) for a in ("NA", "NB", "NC", "ND")]
    c0 = X0[kc].mean(0)
    nrm = np.linalg.svd(X0[kc] - c0)[2][-1]
    if nrm @ (X0[names.index("NE2")] - c0) < 0:
        nrm = -nrm
    edge = np.cross(nrm, X0[names.index("NA")] - X0[names.index("NC")])
    edge /= np.linalg.norm(edge)
    sets = {"crystal": ("crystal.xyz", "<0.55,0.57,0.60>"), "pet-mad-xs": ("pet-mad-xs.xyz", "<0.16,0.47,0.84>"),
            "pet-omol-s-quintet": ("pet-omol-s-quintet.xyz", "<0.92,0.41,0.20>")}
    elr = np.array(["Fe" if a == "FE" else a[0] for a in names])
    for tag, (f, col) in sets.items():
        if not (R / f).exists():
            continue
        X = ase.io.read(R / f).get_positions() / 10
        Rm, pc, qc = kabsch(X[core], X0[core])
        X = (X - pc) @ Rm + qc
        g = X[names.index("FE")]
        n4 = X[kc]
        plane = (g - n4.mean(0)) @ nrm
        FACTS.setdefault("relaxed", {})[tag] = {
            "fe_ne2_nm": float(np.linalg.norm(g - X[names.index("NE2")])),
            "fe_np_nm": float(np.linalg.norm(n4 - g, axis=1).mean()),
            "fe_plane_nm": float(plane)}
        txt = header(edge, c0 + nrm * 0.12, 1.55, ortho=True, dist=40, up=tuple(nrm))
        ids = [k for k in heavy]
        txt += ball_stick(ids, col, r=0.03, rs=0.014, X=X, e=elr, big_fe=0.062)
        # a thin disc marking the pyrrole-nitrogen plane of the crystal
        txt += (f"cylinder{{{P(c0 - nrm * 0.001)},{P(c0 + nrm * 0.001)},0.55 pigment{{rgbt <0.5,0.55,0.6,0.85>}} "
                f"finish{{ambient 0.4 diffuse 0.3}} no_shadow}}\n")
        (OUT / f"relaxed_{tag}.pov").write_text(txt)

FACTS["system"] = {"atoms": int(n), "waters": int(len(sol_o)), "na": int((resn == "NA").sum()), "cl": int((resn == "CL").sum()),
                   "protein_atoms": int(len(protein)), "chains": [int(len(c)) for c in chains], "ml_atoms": int(len(ml))}
(OUT / "facts.json").write_text(json.dumps(FACTS, indent=1))
print(json.dumps(FACTS, indent=1))
