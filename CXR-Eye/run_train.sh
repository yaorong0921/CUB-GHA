python main.py --model_type gat \
                     --dropout 0.0 \
                     --epochs 60 \
                     --lr 0.0005\
                     --step_size 40 \
                     --scheduler \
                     --resize 224 \
                     --gpus 1 \
                     --batch_size 32 \
                     --num_workers 12 \
                     --data_path /storage/rong/CXR-JPG/egd-cxr/1.0.0/master_sheet.csv \
                     --image_path /storage/rong/CXR-JPG/files \
                     --heatmaps_path /storage/rong/CXR-JPG/egd-cxr/fixation_heatmaps \
                     --output_dir ./checkpoint \
                     --viz \
                     --heatmaps_threshold 0.0  \
                     --rseed 1 \
                     --crossval \
                     # --pretrained_dir ./checkpoint/gat_crossv-True_rseed1 \


