
rm -rf ./output
mkdir ./output

python train_c3vg.py \
  --gpu_id=0 \
  --model_name=View_Gen \
  --lr=1e-4 \
  --max_len=500 \
  --max_target_length=300 \
  --epochs=5 \
  --batch_size=12 \
  --save_path=./output/
