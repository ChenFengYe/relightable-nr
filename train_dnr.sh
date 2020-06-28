python train_dnr.py --data_root ./data/material_sphere_cx180 --img_dir _/rgb0 --img_size 512 \
--obj_fp _/mesh.obj \
--texture_num_ch 24 \
--batch_size 2 --gpu_id 1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
--exp_name example \
--checkpoint ./data/material_sphere_cx180/logs/dnr/06-22_05-19-40_skipinv_10_material_sphere_cx180_example/model_epoch-1999_iter-30000.pth \
--start_epoch 2000 --max_epoch 10000

# python train_dnr.py --data_root ./data/material_sphere_cx180 --img_dir _/rgb0 --img_size 512 \
# --obj_fp _/mesh.obj \
# --texture_num_ch 24 \
# --batch_size 1 --gpu_id 1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
# --exp_name example