import sys, hashlib, torch, metatomic.torch, metatensor.torch as mts  # registers the custom classes in the checkpoint
names = ("additive_models.0.energy_composition_buffer", "scaler.energy_scaler_buffer", "scaler.non_conservative_force_scaler_buffer")
sds = {p.split("/")[-1]: torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"] for p in sys.argv[1:]}
for n in names:
    print(n, {k: hashlib.md5(sd[n].numpy().tobytes()).hexdigest()[:8] for k, sd in sds.items()})
for k, sd in sds.items():
    e = mts.load_buffer(sd["scaler.energy_scaler_buffer"])
    c = mts.load_buffer(sd["additive_models.0.energy_composition_buffer"])
    print(k, "energy scale", e.block().values.flatten()[:3].tolist(), "| composition H,C,O,Si:",
          [round(float(v), 4) for v in c.block().values.flatten()[[0, 5, 7, 13]]], "n", c.block().values.numel())
