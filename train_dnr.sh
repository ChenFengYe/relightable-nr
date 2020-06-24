python train_dnr.py --data_root ./data/material_sphere_cx180 --img_dir _/rgb0 --img_size 512 \
--obj_fp _/mesh.obj \
--texture_num_ch 24 \
--batch_size 5 --gpu_id 1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
--exp_name example

# python train_dnr.py --data_root ./data/material_sphere_cx180 --img_dir _/rgb0 --img_size 512 \
# --obj_fp _/mesh.obj \
# --texture_num_ch 24 \
# --batch_size 1 --gpu_id 1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
# --exp_name example