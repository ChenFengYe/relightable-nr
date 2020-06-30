# # # precompute data related to proxy mesh
# python precompute.py --data_root ./data/material_sphere_cx180/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 1 --img_size 512 --only_mesh_related 0 --sampling_pattern all

# python precompute.py --data_root ./data/material_sphere_cx180/ --obj_fp _/mesh_7500v.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 1 --img_size 512 --only_mesh_related 1 --sampling_pattern all

# # # stitch initial environment map from input images
# python stitch_lp.py --data_root ./data/material_sphere_cx180/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --lighting_idx 0 --sampling_pattern skipinv_10 --img_suffix .png

# -----------
# python precompute.py --data_root ./data/material_sphere_gai160/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 0 --img_size 512 --only_mesh_related 0 --sampling_pattern all

# python precompute.py --data_root ./data/material_sphere_gai160/ --obj_fp _/mesh_7500v.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 0 --img_size 512 --only_mesh_related 1 --sampling_pattern all

# # # stitch initial environment map from input images
# python stitch_lp.py --data_root ./data/material_sphere_gai160/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --lighting_idx 0 --sampling_pattern skipinv_10 --img_suffix .png
# -----------

# # python precompute.py --data_root ./data/material_sphere/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 1 --img_size 512 --only_mesh_related 0 --sampling_pattern all

# # python precompute.py --data_root ./data/material_sphere/ --obj_fp _/mesh_7500v.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 1 --img_size 512 --only_mesh_related 1 --sampling_pattern all

# # stitch initial environment map from input images
# python stitch_lp.py --data_root ./data/material_sphere/ --obj_fp _/mesh.obj --calib_fp _/calib.mat --lighting_idx 0 --sampling_pattern skipinv_10 --img_suffix .exr
# -----------

python precompute.py --data_root ./data/synthesis_gai/ --obj_fp _/mesh/%03d.obj --calib_fp _/calib.mat --calib_format convert --img_dir _/rgb0 --gpu_id 1 --img_size 512 --only_mesh_related 0 --sampling_pattern all --multi_frame True

# # stitch initial environment map from input images
#python stitch_lp.py --data_root ./data/synthesis_gai/ --obj_fp _/mesh/%03d.obj --calib_fp _/calib.mat --lighting_idx 0 --sampling_pattern skipinv_10 --img_suffix .png
