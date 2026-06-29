# MD Stress Inputs Needed For Exact Fig. 6

Fig. 6 uses `sigma_c(t)` for convective clusters. The paper says the MD
calculation uses an iso-configurational ensemble (ICE) with 50 replicants for
each cluster. The analysis code therefore expects an NPZ trajectory containing:

```text
per_atom_stress_xy[frame, atom]
```

For LAMMPS, dump the per-atom virial/stress during production. A minimal pattern
is:

```lammps
compute Sxy all stress/atom NULL virial
dump atom_traj all custom ${DUMP_EVERY} dump.shear_${SHEAR_RATE}.lammpstrj &
     id type x y z ix iy iz xu yu zu vx vy vz c_Sxy[4]
```

`compute stress/atom` reports per-atom stress times volume with LAMMPS sign
convention. The Python code can average the corresponding `xy` component over
cluster particles with the particle-volume weights in Eq. (4), but the precise
sign and normalization must be checked against Eq. (2) for the final figure.

The current high-rate production dump does not include `c_Sxy[4]`, so exact
Fig. 6(a,e) cannot be produced from those dumps alone.
