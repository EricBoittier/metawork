# ONIOM campaign summary

21/27 jobs done.

## Means over seeds

| box (nm) | ML region | seeds | atoms | ML atoms | NVE drift (kJ/mol/ps) | NVE residual std (kJ/mol) | T NVE (K) | density NpT (kg/m3) | ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | peptide | 3 | 6486 | 87 ± 0 | -0.44 ± 0.04 | 0.45 ± 0.05 | 300.6 ± 1.8 | 988.4 ± 3.6 | 14.6 ± 0.1 |
| 4.0 | shell | 3 | 6486 | 460 ± 6 | 0.09 ± 0.41 | 2.03 ± 0.50 | 302.3 ± 2.0 | 1037.3 ± 2.6 | 21.7 ± 0.3 |
| 4.0 | shellres | 1 | 6486 | 474 | 0.25 | 1.49 | 303.3 | 1034.3 | 75.6 |
| 5.0 | peptide | 3 | 12162 | 87 ± 0 | -0.98 ± 0.14 | 0.55 ± 0.04 | 299.0 ± 0.2 | 987.4 ± 6.2 | 17.4 ± 0.3 |
| 5.0 | shell | 3 | 12162 | 433 ± 23 | -0.88 ± 0.03 | 1.32 ± 0.09 | 301.1 ± 2.6 | 1005.7 ± 2.1 | 24.1 ± 0.7 |
| 5.0 | shellres | 1 | 12162 | 435 | 0.08 | 1.18 | 302.2 | 1007.7 | 29.2 |
| 6.0 | peptide | 3 | 21084 | 87 ± 0 | -1.68 ± 0.11 | 0.61 ± 0.09 | 299.2 ± 0.9 | 983.0 ± 1.7 | 21.9 ± 0.3 |
| 6.0 | shell | 3 | 21084 | 446 ± 3 | -1.48 ± 0.09 | 2.08 ± 0.44 | 300.6 ± 2.3 | 998.8 ± 2.2 | 29.2 ± 0.2 |
| 6.0 | shellres | 1 | 21084 | 459 | -0.17 | 1.22 | 298.3 | 1000.0 | 52.6 |

## Jobs

Shell kept: ML waters with any atom within 0.5 nm of the peptide at the end of NVE; MM in: MM waters that moved that close; farthest: the shell water farthest from the peptide (nm).

| job | status | ML atoms | drift | std | ring bond (nm) | uncertainty warnings | shell kept | MM in | farthest |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| box5_peptide_s1 | done | 87 | -1.13 | 0.59 | 0.137 | 93 |  |  |  |
| box5_shell_s1 | done | 411 | -0.90 | 1.25 | 0.140 | 844 |  |  |  |
| box4_peptide_s1 | done | 87 | -0.49 | 0.48 | 0.135 | 24 |  |  |  |
| box6_peptide_s1 | done | 87 | -1.81 | 0.70 | 0.136 | 9 |  |  |  |
| box4_shell_s1 | done | 462 | 0.23 | 1.84 | 0.139 | 418 |  |  |  |
| box6_shell_s1 | done | 450 | -1.44 | 1.96 | 0.140 | 102 |  |  |  |
| box5_peptide_s2 | done | 87 | -0.93 | 0.54 | 0.134 | 4 |  |  |  |
| box5_shell_s2 | done | 456 | -0.85 | 1.28 | 0.135 | 592 |  |  |  |
| box4_peptide_s2 | done | 87 | -0.42 | 0.39 | 0.135 | 14 |  |  |  |
| box6_peptide_s2 | done | 87 | -1.64 | 0.62 | 0.134 | 17 |  |  |  |
| box4_shell_s2 | done | 465 | -0.37 | 1.66 | 0.138 | 874 |  |  |  |
| box6_shell_s2 | done | 444 | -1.42 | 2.57 | 0.138 | 15 |  |  |  |
| box5_peptide_s3 | done | 87 | -0.87 | 0.52 | 0.140 | 33 |  |  |  |
| box5_shell_s3 | done | 432 | -0.89 | 1.42 | 0.135 | 343 |  |  |  |
| box4_peptide_s3 | done | 87 | -0.41 | 0.49 | 0.132 | 2 |  |  |  |
| box6_peptide_s3 | done | 87 | -1.60 | 0.52 | 0.139 | 2 |  |  |  |
| box4_shell_s3 | done | 453 | 0.41 | 2.59 | 0.136 | 307 |  |  |  |
| box6_shell_s3 | done | 444 | -1.58 | 1.71 | 0.138 | 869 |  |  |  |
| box5_shellres_s1 | done | 435 | 0.08 | 1.18 | 0.133 | 219 | 74/116 | 91 | 0.82 |
| box4_shellres_s1 | done | 474 | 0.25 | 1.49 | 0.137 | 77 | 99/129 | 88 | 0.75 |
| box6_shellres_s1 | done | 459 | -0.17 | 1.22 | 0.136 | 166 | 88/124 | 89 | 0.75 |
| box5_shellres_s2 | pending (tries 1, done: build,em,index,npt,nvt) |  |  |  |  |  |  |  |  |
| box4_shellres_s2 | pending (tries 0, done: -) |  |  |  |  |  |  |  |  |
| box6_shellres_s2 | pending (tries 0, done: -) |  |  |  |  |  |  |  |  |
| box5_shellres_s3 | pending (tries 0, done: -) |  |  |  |  |  |  |  |  |
| box4_shellres_s3 | pending (tries 0, done: -) |  |  |  |  |  |  |  |  |
| box6_shellres_s3 | pending (tries 0, done: -) |  |  |  |  |  |  |  |  |
