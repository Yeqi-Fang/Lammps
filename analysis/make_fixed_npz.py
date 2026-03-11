import numpy as np
from fix_cage_jump_positions import extract_real_positions_fixed

npz_in  = "cage_jumps_shearrate_0p015.npz"
dumpf   = "dump.shear_0.015.lammpstrj"
npz_out = "cage_jumps_shearrate_0p015_FIXED.npz"

data = np.load(npz_in, allow_pickle=True)

jumps = {
    "particle_idx": data["particle_idx"],
    "jump_frames":  data["jump_frames"],
    "jump_vectors": data["jump_vectors"],
}
xy, valid = extract_real_positions_fixed(jumps, dumpf)   # 会自动把 xu/yu 折回盒内

# 把原 npz 里所有字段拷贝出来，只替换 jump_pos_xy
out_dict = {k: data[k] for k in data.files if k != "jump_pos_xy"}
out_dict["jump_pos_xy"] = xy
np.savez(npz_out, **out_dict)

print("saved:", npz_out, "jump_pos_xy:", xy.shape)