python tools/test_dnr.py --cfg experiments/dataset_name/net_name/dnr_dome_512_lr1e-3.yaml

# novel view synthesis
# python tools/test_dnr.py \
# --calib_dir _/test_calib53 \
# --calib_name calib20200619_test_mid_53.mat \
# --img_size 512 --sampling_pattern all --gpu_id 0 \
# --checkpoint_dir ./data/synthesis_gai/logs/dnr/07-01_10-01-45_skipinv_10_synthesis_gai_example \
# --checkpoint_name model_epoch-372_iter-32000.pth \
# --save_folder img_est_350
#--checkpoint_dir ./data/synthesis_gai/logs/dnr/07-01_10-01-45_skipinv_10_synthesis_gai_example \
#--checkpoint_name model_epoch-58_iter-5000.pth
#--checkpoint_dir ./data/synthesis_gai/logs/dnr/06-30_11-06-54_skipinv_10_synthesis_gai_example \
#--checkpoint_name model_epoch-541_iter-105000.pth
# --checkpoint_dir ./data/synthesis_gai/logs/dnr/07-01_10-01-45_skipinv_10_synthesis_gai_example \
# --checkpoint_name model_epoch-232_iter-20000.pth \
