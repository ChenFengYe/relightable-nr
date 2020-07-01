# novel view synthesis
python test_dnr.py --calib_dir _/test_seq/mid_step360 \
--checkpoint_dir ./data/synthesis_gai/logs/dnr/06-30_11-06-54_skipinv_10_synthesis_gai_example \
--checkpoint_name model_epoch-531_iter-105000.pth \
--img_size 512 --sampling_pattern all --gpu_id 2 \
--multi_frame True
