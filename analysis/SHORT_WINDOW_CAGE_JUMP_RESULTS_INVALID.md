# Invalid short-window cage-jump results

These outputs are retained only for audit history. They must not be used for
Zeng Fig. 5/Fig. 6 reproduction.

Reason: Candelier cage-jump detection is recursively path-dependent and must be
performed on a sufficiently long trajectory before selecting events in a
`t_chi` window. Direct detection on a 31-32 frame window inflates the event
count by roughly one order of magnitude.

Checked examples for type 1 at gamma_dot = 0.015:

- Long 1500-frame trajectory, then select frames 0-30: 628 events.
- Direct 31-frame detection: 5697 events.
- Direct 32-frame detection: 6014 events.

Invalid input/result directories:

- `analysis/cage_jumps_md3d_gdot0p015_window31_type1`
- `analysis/cage_jumps_md3d_gdot0p015_window31_type2`
- `analysis/cage_jumps_md3d_gdot0p015_window31_unwrapped`
- `analysis/cage_jumps_md3d_gdot0p015_tchi31_type1_audit`
- `analysis/cage_jumps_md3d_gdot0p015_tchi31_type1_minseg10_audit`
- `analysis/cluster_md3d_gdot0p015_window31_type1`
- `analysis/cluster_md3d_gdot0p015_window31_type2`
- `analysis/cluster_md3d_gdot0p015_window31_unwrapped`
- `analysis/cluster_md3d_gdot0p015_tchi31_type1_audit`
- `analysis/cluster_md3d_gdot0p015_tchi31_type1_audit_pbc`
- `analysis/cluster_md3d_gdot0p015_tchi31_type1_audit_pbc_diag`

Formal chain from this point:

1. Use type-1 chi4, `t_chi = 2.4 tau0`.
2. Use long-trajectory cage jumps only:
   `analysis/cage_jumps_md3d_gdot0p015_1500f_type1_wrapped/trial_cage_jumps.npz`.
3. Select cage-jump events inside each `t_chi` window after detection.
4. Build clusters from those window-selected events.
