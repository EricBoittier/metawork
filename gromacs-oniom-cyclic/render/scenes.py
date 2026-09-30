"""Write POV-Ray scenes of the ONIOM campaign systems and print measured numbers.

Coordinates come from the job .gro files; the ML region from each job's ml.ndx.
GROMACS (x, y, z) maps to POV <x, z, y> (z up), which also fixes handedness.
"""
import json, sys
from pathlib import Path
import numpy as np

JOBS = Path("/home/boittier/metawork/gromacs-oniom-cyclic/campaign/jobs")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT.mkdir(parents=True, exist_ok=True)
FACTS = {}


def read_gro(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[1])
    resid, resn, name, xyz = [], [], [], []
    for l in lines[2:2 + n]:
        resid.append(int(l[0:5])); resn.append(l[5:10].strip()); name.append(l[10:15].strip())
        xyz.append([float(l[20:28]), float(l[28:36]), float(l[36:44])])
    box = np.array([float(v) for v in lines[2 + n].split()[:3]])
    return dict(resid=np.array(resid), resn=np.array(resn), name=np.array(name), xyz=np.array(xyz), box=box)


def read_ndx(path):
    return np.array([int(t) - 1 for l in Path(path).read_text().splitlines()[1:] for t in l.split()])


def mic(d, box):
    return d - box * np.round(d / box)


def elem(n):
    return "H" if n.startswith("H") else n[0]


def prepare(gro):
    """Make the peptide whole, put its centroid at the origin, wrap everything else around it."""
    s = read_gro(gro)
    xyz, box = s["xyz"].copy(), s["box"]
    pep = np.where(s["resn"] != "SOL")[0]
    # peptide: unwrap every atom against atom 0 (the peptide is < half the box)
    xyz[pep] = xyz[pep[0]] + mic(xyz[pep] - xyz[pep[0]], box)
    c = xyz[pep].mean(0)
    xyz = c + mic(xyz - c, box)
    xyz[pep] = xyz[pep[0]] + mic(xyz[pep] - xyz[pep[0]], box)
    # waters: keep H with their O
    sol = np.where(s["name"] == "OW")[0]
    for h in (1, 2):
        xyz[sol + h] = xyz[sol] + mic(xyz[sol + h] - xyz[sol], box)
    s["xyz"] = xyz - c
    s["pep"] = pep
    s["el"] = np.array([elem(n) for n in s["name"]])
    return s


def bonds(xyz, el, idx):
    b = []
    for a in range(len(idx)):
        for k in range(a + 1, len(idx)):
            i, j = idx[a], idx[k]
            d = np.linalg.norm(xyz[i] - xyz[j])
            lim = 0.125 if "H" in (el[i], el[j]) else 0.175
            if el[i] == "H" and el[j] == "H":
                continue
            if d < lim:
                b.append((i, j))
    return b


def P(v):
    return f"<{v[0]:.4f},{v[2]:.4f},{v[1]:.4f}>"


COL = {"C": "<0.50,0.52,0.55>", "N": "<0.20,0.38,0.90>", "O": "<0.90,0.20,0.15>", "H": "<0.93,0.93,0.93>"}
RES_C = {1: "<0.70,0.70,0.70>", 2: "<0.12,0.66,0.62>", 3: "<0.52,0.76,0.25>",
         4: "<0.90,0.40,0.62>", 5: "<0.62,0.42,0.88>", 6: "<0.66,0.48,0.30>"}
MM_W = "<0.55,0.72,0.88>"
ML_W = "<0.98,0.55,0.12>"


def header(cam_dir, look, width, aspect=1.0, ortho=True, dist=40, up=(0, 0, 1)):
    cd = np.array(cam_dir, float); cd /= np.linalg.norm(cd)
    loc = np.array(look) + cd * dist
    proj = "orthographic" if ortho else "perspective"
    ang = f"angle {np.degrees(2*np.arctan(width/(2*dist))):.5f}" if ortho else f"angle {width}"
    right_up = f"right x*{aspect:.3f} up y"
    return f"""#version 3.7;
global_settings {{ assumed_gamma 1.0 max_trace_level 12 }}
background {{ rgbt <1,1,1,1> }}
camera {{ {proj} location {P(loc)} {right_up} sky {P(up)} {ang} look_at {P(look)} }}
light_source {{ {P(np.array(look) + 60*(cd + np.array([-0.6, 0.4, 1.2])))} rgb 1.0 area_light <4,0,0>,<0,4,0>,4,4 adaptive 1 jitter }}
light_source {{ {P(np.array(look) + 60*(cd + np.array([0.8, -0.8, -0.2])))} rgb 0.35 shadowless }}
#declare F = finish {{ ambient 0.22 diffuse 0.72 specular 0.35 roughness 0.02 }};
#declare FW = finish {{ ambient 0.30 diffuse 0.65 specular 0.15 roughness 0.05 }};
"""


def sph(p, r, col, t=0.0, fin="F", shadow=True):
    ns = "" if shadow else " no_shadow"
    return f"sphere{{{P(p)},{r:.4f} pigment{{rgbt <{col.strip('<>')},{t:.2f}>}} finish{{{fin}}}{ns}}}\n"


def cyl(a, b, r, col, t=0.0, fin="F", shadow=True):
    if np.linalg.norm(np.asarray(a) - np.asarray(b)) < 1e-6:
        return ""
    ns = "" if shadow else " no_shadow"
    return f"cylinder{{{P(a)},{P(b)},{r:.4f} pigment{{rgbt <{col.strip('<>')},{t:.2f}>}} finish{{{fin}}}{ns}}}\n"


def peptide(s, style="bs", res_color=False, scale=1.0, ring=True, bl=None):
    xyz, el, pep = s["xyz"], s["el"], s["pep"]
    bl = bl if bl is not None else bonds(xyz, el, pep)
    out = []
    rad = {"bs": {"H": 0.022, "C": 0.038, "N": 0.038, "O": 0.038},
           "cpk": {"H": 0.075, "C": 0.13, "N": 0.12, "O": 0.12}}[style]
    def c_of(i):
        if res_color and el[i] == "C":
            return RES_C[s["resid"][i]]
        return COL[el[i]]
    for i in pep:
        out.append(sph(xyz[i], rad[el[i]] * scale, c_of(i)))
    if style == "bs":
        for i, j in bl:
            m = (xyz[i] + xyz[j]) / 2
            out.append(cyl(xyz[i], m, 0.018 * scale, c_of(i)))
            out.append(cyl(m, xyz[j], 0.018 * scale, c_of(j)))
    if ring:
        i, j = ring_pair(s)
        out.append(cyl(xyz[i], xyz[j], 0.030 * scale, "<1.0,0.84,0.0>"))
    return "".join(out)


def ring_pair(s):
    n, r, name = len(s["pep"]), s["resid"], s["name"]
    c6 = np.where((r == 6) & (name == "C"))[0][0]
    n1 = np.where((r == 1) & (name == "N"))[0][0]
    return c6, n1


def waters(s, idx_o, col, t=0.0, scale=1.0, shadow=True, fin="FW"):
    xyz = s["xyz"]
    out = []
    for o in idx_o:
        out.append(sph(xyz[o], 0.075 * scale, col, t, fin, shadow))
        for h in (1, 2):
            out.append(sph(xyz[o + h], 0.045 * scale, col, t, fin, shadow))
            out.append(cyl(xyz[o], xyz[o + h], 0.025 * scale, col, t, fin, shadow))
    return "".join(out)


def box_frame(L, col="<0.35,0.38,0.42>", r=0.018):
    h = np.array(L) / 2
    corners = [np.array([sx, sy, sz]) * h for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    out = []
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b]) > 1e-9) == 1:
                out.append(cyl(corners[a], corners[b], r, col))
    out += [sph(c, r, col) for c in corners]
    return "".join(out)


def ring_torus(center, radius, normal, col, r=0.012):
    n = np.array(normal, float); n /= np.linalg.norm(n)
    # POV torus lies in the xz plane (POV coords) = GROMACS xy plane, axis GROMACS z
    zg = np.array([0, 0, 1.0])
    ax = np.cross(zg, n); s_ = np.linalg.norm(ax); cth = zg @ n
    # rotation matrix GROMACS frame, then permute to POV frame
    if s_ < 1e-9:
        R = np.eye(3)
    else:
        ax /= s_; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + s_ * K + (1 - cth) * K @ K
    Pm = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    M = Pm @ R @ Pm
    m = " ".join(f"{v:.5f}," for v in M.T.flatten())
    return (f"torus{{{radius:.4f},{r:.4f} pigment{{rgb {col}}} finish{{F}} no_shadow "
            f"matrix<{m} 0,0,0> translate {P(center)}}}\n")


cam = np.array([1.0, -1.35, 0.8])  # GROMACS-frame view direction used throughout


def cutaway_sel(s, direction, keep_all=()):
    """Waters (O index) outside the octant that faces the camera."""
    sol = np.where(s["name"] == "OW")[0]
    d = np.sign(direction)
    p = s["xyz"][sol]
    front = (p[:, 0] * d[0] > 0) & (p[:, 1] * d[1] > 0) & (p[:, 2] > -0.6)
    return sol[~front | np.isin(sol, keep_all)]


def behind(s, ids, off=0.0):
    c = cam / np.linalg.norm(cam)
    return ids[s["xyz"][ids] @ c < off]


def write(name, text):
    (OUT / f"{name}.pov").write_text(text)


# ---------------------------------------------------------------- 1. box sizes
sysfacts = {}
for b in (4, 5, 6):
    job = JOBS / f"box{b}_shell_s1"
    s = prepare(job / "nvt.gro")
    ml = read_ndx(job / "ml.ndx")
    ml_o = ml[s["name"][ml] == "OW"]
    keep = cutaway_sel(s, cam)
    mm_o = np.setdiff1d(keep, ml_o)
    shown_ml = np.intersect1d(ml_o, keep)
    txt = header(cam, (0, 0, 0), 10.6, ortho=True, dist=60)
    txt += box_frame(s["box"])
    # depth cue: nearer waters darker steel blue, farther ones paler
    depth = s["xyz"][mm_o] @ (cam / np.linalg.norm(cam))
    q = (depth - depth.min()) / (np.ptp(depth) + 1e-9)
    near_c, far_c = np.array([0.30, 0.47, 0.66]), np.array([0.86, 0.91, 0.96])
    for bin_ in range(8):
        sel = mm_o[(np.minimum((q * 8).astype(int), 7)) == bin_]
        c = far_c + (near_c - far_c) * (bin_ + 0.5) / 8
        txt += waters(s, sel, f"<{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}>", 0.0, 1.0, shadow=False)
    txt += waters(s, ml_o, ML_W, 0.0, 1.0)
    txt += peptide(s, "cpk", res_color=False, ring=False)
    write(f"box{b}", txt)
    sysfacts[b] = dict(atoms=int(len(s["xyz"])), waters=int((s["name"] == "OW").sum()),
                       box_nvt=float(s["box"][0]), ml_waters=int(len(ml_o)), ml_atoms=int(len(ml)))
FACTS["boxes"] = sysfacts

# ---------------------------------------------------------------- shared close-up frame
job = JOBS / "box5_shell_s1"
s = prepare(job / "nvt.gro")
ml = read_ndx(job / "ml.ndx")
ml_o = ml[s["name"][ml] == "OW"]
pep = s["pep"]
bl = bonds(s["xyz"], s["el"], pep)
sol = np.where(s["name"] == "OW")[0]
xyz = s["xyz"]


def dmin_to_pep(ids, X=None):
    X = xyz if X is None else X
    return np.array([np.min(np.linalg.norm(X[pep] - X[i], axis=1)) for i in ids])


# ---------------------------------------------------------------- 2. peptide hero, residue colours
txt = header(cam, (0, 0, 0), 1.75, ortho=True)
txt += peptide(s, "bs", res_color=True, scale=1.25, bl=bl)
write("peptide", txt)
c6, n1 = ring_pair(s)
FACTS["ring_bond_nvt_nm"] = float(np.linalg.norm(xyz[c6] - xyz[n1]))

# ---------------------------------------------------------------- 3. ML region: peptide only vs shell
near = sol[dmin_to_pep(sol) < 0.95]
slab = behind(s, near, 0.15)  # waters near the peptide, front half cut away
for tag, mlset in (("region_peptide", np.array([], int)), ("region_shell", ml_o)):
    txt = header(cam, (0, 0, 0), 3.4, ortho=True)
    txt += waters(s, np.setdiff1d(slab, mlset), MM_W, 0.55, shadow=False)
    txt += waters(s, np.intersect1d(slab, mlset), ML_W, 0.0)
    txt += peptide(s, "bs", bl=bl, ring=False)
    write(tag, txt)
FACTS["shell_waters_box5_s1"] = int(len(ml_o))

# ---------------------------------------------------------------- 4. pairs beyond rcoulomb
D = np.linalg.norm(xyz[pep][:, None] - xyz[pep][None], axis=2)
iu = np.triu_indices(len(pep), 1)
far = [(pep[a], pep[b]) for a, b in zip(*iu) if D[a, b] >= 1.0]
FACTS["pep_pairs_total"] = int(len(iu[0]))
FACTS["pep_pairs_beyond_rc"] = int(len(far))
a, b = np.unravel_index(np.argmax(D), D.shape)
FACTS["pep_max_width_nm"] = float(D[a, b])
FACTS["pep_widest_pair"] = [f"{s['resid'][pep[a]]}{s['resn'][pep[a]]}-{s['name'][pep[a]]}",
                            f"{s['resid'][pep[b]]}{s['resn'][pep[b]]}-{s['name'][pep[b]]}"]
txt = header(cam, 0.55 * xyz[pep[a]], 2.5, ortho=True)
txt += peptide(s, "bs", bl=bl, ring=False)
txt += ring_torus(xyz[pep[a]], 1.0, cam, "<0.16,0.42,0.86>", 0.012)
for i, j in far:
    txt += cyl(xyz[i], xyz[j], 0.0045, "<0.85,0.15,0.15>", 0.55, shadow=False)
A = xyz[pep[a]]
txt += cyl(xyz[pep[a]], xyz[pep[b]], 0.012, "<0.75,0.05,0.05>", 0.0)
txt += sph(xyz[pep[a]], 0.07, "<0.75,0.05,0.05>") + sph(xyz[pep[b]], 0.07, "<0.75,0.05,0.05>")
write("param_rc_pairs", txt)

# ---------------------------------------------------------------- 5. cutoff rings: rc 1.0, PET-MAD 1.5, rlist 1.55
cd = cam / np.linalg.norm(cam)
wide = behind(s, sol[dmin_to_pep(sol) < 1.75], 0.0)
txt = header(cam, (0, 0, 0), 4.6, ortho=True, dist=60)
txt += waters(s, np.setdiff1d(wide, ml_o), MM_W, 0.72, shadow=False)
txt += peptide(s, "bs", bl=bl, ring=False)
cen = xyz[pep[a]]
txt += sph(cen, 0.08, "<0.10,0.10,0.10>")
for rr, col in ((1.0, "<0.16,0.42,0.86>"), (1.5, "<0.92,0.40,0.14>"), (1.55, "<0.10,0.62,0.42>")):
    txt += ring_torus(cen, rr, cd, col, 0.018)
write("param_cutoffs", txt)
FACTS["cutoff_center_atom"] = FACTS["pep_widest_pair"][0]

# ---------------------------------------------------------------- 6. the 0.5 nm shell rule
txt = header(cam, (0, 0, 0), 3.0, ortho=True)
env = "intersection{ merge{\n" + "".join(f"sphere{{{P(xyz[i])},0.5}}\n" for i in pep) + \
      "} plane{" + P(cam / np.linalg.norm(cam)) + ", 0.05} pigment{rgbt <0.98,0.62,0.20,0.80>} finish{ambient 0.35 diffuse 0.5 specular 0.2} no_shadow}\n"
txt += env
dm = dmin_to_pep(sol)
just_out = sol[(dm >= 0.5) & (dm < 0.8)]
txt += waters(s, behind(s, just_out, 0.05), MM_W, 0.35, shadow=False)
txt += waters(s, behind(s, ml_o, 0.05), ML_W, 0.0)
txt += peptide(s, "bs", bl=bl, ring=False)
write("param_shell", txt)

# ---------------------------------------------------------------- 7. shell waters at start vs end of the ONIOM run
e = prepare(job / "nve.gro")
# align the end frame on the peptide (Kabsch) so the view is the same
def kabsch(Pm, Q):
    Pc, Qc = Pm - Pm.mean(0), Q - Q.mean(0)
    U, S_, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d])
    return U @ Dm @ Vt, Pm.mean(0), Q.mean(0)
R, pc, qc = kabsch(e["xyz"][pep], xyz[pep])
e["xyz"] = (e["xyz"] - pc) @ R + qc
de = np.array([np.min(np.linalg.norm(e["xyz"][pep] - e["xyz"][i], axis=1)) for i in ml_o])
d0 = dm[np.searchsorted(sol, ml_o)]
FACTS["shell_start_max_dist"] = float(d0.max())
FACTS["shell_end_within_0p5"] = int((de < 0.5).sum())
FACTS["shell_end_beyond_1"] = int((de >= 1.0).sum())
FACTS["shell_end_median_dist"] = float(np.median(de))
FACTS["shell_end_max_dist"] = float(de.max())
FACTS["shell_total"] = int(len(ml_o))
FACTS["nve_end_new_waters_within_0p5"] = int((np.array([np.min(np.linalg.norm(e["xyz"][pep] - e["xyz"][i], axis=1)) for i in np.setdiff1d(sol, ml_o)]) < 0.5).sum())
for tag, S in (("shell_start", s), ("shell_end", e)):
    txt = header(cam, (0, 0, 0), 5.2, ortho=True, dist=60)
    txt += box_frame(S["box"], r=0.012) if False else ""
    txt += waters(S, ml_o, ML_W, 0.0)
    mm = np.setdiff1d(sol, ml_o)
    dmm = np.array([np.min(np.linalg.norm(S["xyz"][pep] - S["xyz"][i], axis=1)) for i in mm])
    txt += waters(S, mm[dmm < 0.5], "<0.16,0.42,0.86>", 0.0)
    txt += peptide(S, "bs", bl=bl, ring=False)
    write(tag, txt)

# ---------------------------------------------------------------- 8. three seeds overlaid (end of NVE)
cols = ["<0.16,0.47,0.84>", "<0.92,0.41,0.20>", "<0.11,0.69,0.48>"]
ref = None
txt = header(cam, (0, 0, 0), 2.1, ortho=True)
rmsd = {}
heavy = pep[s["el"][pep] != "H"]
for k, seed in enumerate((1, 2, 3)):
    S = prepare(JOBS / f"box5_peptide_s{seed}" / "nve.gro")
    R, pc, qc = kabsch(S["xyz"][heavy], xyz[heavy])
    X = (S["xyz"] - pc) @ R + qc
    rmsd[seed] = float(np.sqrt(np.mean(np.sum((X[heavy] - xyz[heavy]) ** 2, 1))))
    bh = [(i, j) for i, j in bl if S["el"][i] != "H" and S["el"][j] != "H"]
    for i in heavy:
        txt += sph(X[i], 0.04, cols[k])
    for i, j in bh:
        txt += cyl(X[i], X[j], 0.02, cols[k])
write("seeds", txt)
FACTS["seed_heavy_rmsd_vs_box5_shell_s1_nvt_nm"] = rmsd

(OUT / "facts.json").write_text(json.dumps(FACTS, indent=1))
print(json.dumps(FACTS, indent=1))
