# novel view synthesis
python test_dnr.py --calib_dir _/test_seq/mid_step360 \
--checkpoint_dir ./data/material_sphere_cx180/logs/dnr/06-22_05-19-40_skipinv_10_material_sphere_cx180_example \
--checkpoint_name model_epoch-1999_iter-30000.pth \
--img_size 512 --sampling_pattern all --gpu_id 0 \
--multi_frame True
