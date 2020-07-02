python tools/train_dnr.py --data_root ./data/synthesis_gai --img_dir _/rgb0 --img_size 512 \
--obj_fp _/mesh/%03d.obj \
--texture_num_ch 24 \
--batch_size 2 --gpu_id 0 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
--exp_name example \
--start_epoch 0 --max_epoch 10000 \
--multi_frame True \
--texture_size 1024 \
#--checkpoint ./data/synthesis_gai/logs/dnr/06-30_04-08-01_skipinv_10_synthesis_gai_example/model_epoch-430_iter-87000.pth \

# orig
# python train_dnr.py --data_root ./data/material_sphere_cx180 --img_dir _/rgb0 --img_size 512 \
# --obj_fp _/mesh.obj \
# --texture_num_ch 24 \
# --batch_size 1 --gpu_id 1 --sampling_pattern skipinv_10 --sampling_pattern_val skip_10 --val_freq 100 \
# --exp_name example