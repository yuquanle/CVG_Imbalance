#export CUDA_VISIBLE_DEVICES=3
# rm -rf ./output/
# mkdir ./output/

# bart-base-chinese, mengzi-t5-base, mt5base, Randeng-T5-784M
backbone_model_name=Randeng-T5-784M
# vanilla、label_cond
model_name=label_cond
tensorboard_summary_log_path=/home/leyuquan/projects/LLMs/CVG/criminal_cvg_plm/logs/tensorboard_summary_logs/${backbone_model_name}_${model_name} 

CUDA_VISIBLE_DEVICES=0 python -u ../train_criminal_cvg.py \
  --model_name=${model_name} \
  --lr=1e-4 \
  --epochs=5 \
  --batch_size=1 \
  --accumulation_steps=32 \
  --max_input_length=800 \
  --max_target_length=200 \
  --backbone_model_name=${backbone_model_name} \
  --save_path=./output/${backbone_model_name}_${model_name}_best.pth \
  --tensorboard_summary_log_path=${tensorboard_summary_log_path}
