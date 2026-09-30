# POV-Ray renders

`scenes.py OUTDIR` writes POV-Ray 3.7 scenes of the peptide campaign (box sizes, ML regions,
cut-offs, the 0.5 nm shell rule, shell waters at the start and end, seeds) from the job
`.gro` files; `povlib.py` has the shared helpers (GROMACS xyz to POV `<x,z,y>`, spheres,
cylinders, cameras). Render each scene with

    povray -D +Iscene.pov +Oscene.png +W1400 +H1400 +A0.15 +AM2 +R3 +UA +FN

POV-Ray 3.7 from conda-forge needs an explicit `angle` for orthographic cameras, which
`povlib.header()` sets from the view width.
