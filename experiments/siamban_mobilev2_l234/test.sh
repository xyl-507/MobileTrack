START=10
END=20
for i in $(seq $START $END)
do
    python -u ../../tools/test.py \
        --snapshot /home/xyl/siamban/experiments/siamban_mobilev2_l234/xyl/17.siamban+mobilenetv2-WH-ECA-model.pth/checkpoint_e$i.pth \
	      --config config.yaml \
	      --gpu_id $(($i % 2)) \
	      --dataset UAVDT 2>&1 | tee logs/test_dataset.log &
done
python ../../tools/eval.py --tracker_path ./results --dataset UAVDT --num 4 --tracker_prefix 'ch*'

# learn from siammask
