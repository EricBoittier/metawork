"""POV-Ray scene helpers shared by the render scripts (GROMACS xyz -> POV <x,z,y>)."""
import numpy as np

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


