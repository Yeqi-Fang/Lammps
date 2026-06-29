# Zeng 2025 Reproduction Workspace

This directory is intentionally separate from the older `analysis/` scripts.

Current active scope: **BD documentation and audit only**. No new BD
processing code is added in this phase.

The BD strict-reproduction references are:

- `BD_STRICT_REPRODUCTION.md`: paper-traceable BD route, figure scope,
  missing inputs, and acceptance criteria.
- `configs/BD/bd_parameter_audit.csv`: parameter-by-parameter source audit.
- `ASSUMPTIONS_TO_REVIEW.md`: missing BD literature, metadata, and explicit
  assumptions that must not be presented as paper facts.

Existing scripts in this directory were prepared earlier for broader
paper-traceable workflows. For the current BD-only phase, only the BD entries
below are in scope, and they should not be treated as strict reproduction until
the BD audit file's missing items are resolved:

- BD Fig. 2: convective cluster analysis for `phi=45%`, `gdot*tau0=3`.
- BD Fig. 4: PHM/STZ and cage-jump pair-correlation analysis for the same BD cluster.

The existing scripts should not invent missing physical quantities. If
per-particle stress, forces, PHM modes, or BD trajectories are absent, the
workflow must stop with a clear missing-input message.

## Folder Layout

```text
zeng_reproduction/
  configs/
    BD/
    MD/   # legacy/not active in the current BD-only phase
  data/
    BD/{raw,processed,figures}
    MD/3D/{raw,processed,figures}  # legacy/not active now
    MD/2D/{raw,processed,figures}  # legacy/not active now
  scripts/
  src/zeng_repro/
```

## Expected NPZ Inputs

The code accepts NPZ files with these arrays:

- `positions`: shape `(frames, particles, dim)`, lab-frame coordinates.
- `nonaffine`: shape `(frames, particles, dim)`, Yamamoto-Onuki nonaffine trajectories.
- `times`: shape `(frames,)`.
- `types`: shape `(particles,)`, type `1=big`, `2=small`.
- `box`: shape `(dim,)`, box lengths.
- `jump_frames`, `particle_idx`, `jump_vectors`, `jump_positions`, `jump_times`.
- Optional for exact stress: `per_atom_stress_xy`, shape `(frames, particles)`.
- Optional for Fig. 4(a): `phm_eigenvectors`, `phm_eigenvalues`, `phm_particle_ids`.

Existing `analysis/cage_jump_detection.py` outputs can be converted by the
helper loaders, but exact Fig. 2(a) stress still requires per-particle stress or
force-level data.

## Example Commands

```bash
python zeng_reproduction/scripts/run_fig2_bd.py \
  --config zeng_reproduction/configs/BD/fig2_phi45_gdot3.json

python zeng_reproduction/scripts/run_fig4_bd.py \
  --config zeng_reproduction/configs/BD/fig4_phi45_gdot3.json
```

## Paper-Required Conditions

The constants in `src/zeng_repro/paper_constants.py` are copied from the paper:

- `lc^2`: BD `0.05`, 3D MD `0.057`, 2D MD `0.048`.
- overlap cutoff `a`: BD `0.2`, 3D MD `0.25`, 2D MD `0.3`.
- BD Fig. 2/Fig. 4: `phi=45%`, `gdot*tau0=3`.
- MD constants and configs may exist from earlier work, but they are not part
  of the current BD-only implementation.

See `ASSUMPTIONS_TO_REVIEW.md` and `configs/BD/bd_parameter_audit.csv` for
every BD parameter that is missing, inferred, or requires literature/user
confirmation.
