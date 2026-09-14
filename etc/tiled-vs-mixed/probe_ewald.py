import os, pickle, numpy as np
import bench

bench.select_config("mad")
with open(os.environ["SCRATCH"] + "/tvm_pool.pkl", "rb") as f:
    samples = pickle.load(f)["samples"]

na, srp, lrp, nk, sme, pbcs, cellL = [], [], [], [], [], [], []
for s in samples:
    st, lst = s.structure, s.lr_structure
    na.append(int(st["atom_mask"].sum()))
    srp.append(int(st["pair_mask"].sum()))
    lrp.append(len(lst["centers"]))
    p = int(np.asarray(lst["pbc"]).sum())
    pbcs.append(p)
    if p:
        nk.append(int(lst["lr"].k_grid.shape[0]))
        sme.append(float(lst["smearing"]))
        cellL.append(float(np.linalg.norm(lst["cell"], axis=-1).mean()))

na, srp, lrp = map(np.array, (na, srp, lrp))
nk, sme, cellL = map(np.array, (nk, sme, cellL))
per = np.array(pbcs) > 0

print("pool: %d samples, %d periodic" % (len(na), per.sum()))
print("atoms/structure      mean %7.1f  max %6d" % (na.mean(), na.max()))
print("SR pairs/atom        mean %7.1f" % (srp.sum() / na.sum()))
print("Ewald pairs/atom (periodic only) mean %7.1f" % (lrp[per].sum() / na[per].sum()))
print("Ewald pairs/structure (periodic) mean %7.1f max %d" % (lrp[per].mean(), lrp[per].max()))
print("num_k per periodic structure     mean %7.1f min %d max %d" % (nk.mean(), nk.min(), nk.max()))
print("smearing (A)                     mean %7.3f min %.3f max %.3f" % (sme.mean(), sme.min(), sme.max()))
print("mean cell vector length (A)      mean %7.2f min %.2f max %.2f" % (cellL.mean(), cellL.min(), cellL.max()))
print("implied Ewald real cutoff = 8*lr_wavelength = 4*smearing: mean %.2f A" % (4 * sme.mean()))
print("\nproduction lr_MAD_v16 budget for 32 structures: n_atoms 1024, sr.n_pairs 131072,")
print("  lr.n_pairs 1048576, lr.num_k 5632, k_sel 73")
print("measured here, scaled to 1024 atoms: sr.n_pairs %.0f, lr.n_pairs %.0f, num_k %.0f" % (
    1024 * srp.sum() / na.sum(), 1024 * lrp.sum() / na.sum(), nk.max()))
