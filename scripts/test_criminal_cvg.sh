#export CUDA_VISIBLE_DEVICES=3
# rm -rf ./output/
# mkdir ./output/
# bart-base-chinese, mengzi-t5-base, mt5base, Randeng-T5-784M
backbone_model_name=mengzi-t5-base
# view=200，fact=800
# vanilla, label_cond
model_name=vanilla
CUDA_VISIBLE_DEVICES=0 python -u ../main_criminal_cvg.py \
  --model_name=${model_name} \
  --batch_size=32 \
  --max_input_length=800 \
  --max_target_length=200 \
  --backbone_model_name=${backbone_model_name} \
  --model_path=./output/${backbone_model_name}_${model_name}_best.pth
