#export CUDA_VISIBLE_DEVICES=3
# rm -rf ./output/
# mkdir ./output/
# bart-base-chinese, mengzi-t5-base, mt5base, Randeng-T5-784M
# vanilla, label_cond
model_name=label_cond
backbone_model_name=Randeng-T5-784M
input_pred_result_path=/home/leyuquan/projects/LLMs/CVG/criminal_cvg_plm/output/test_pred_results/${backbone_model_name}_${model_name}_test_result_200.json
test_path=/mnt/sdb/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_test.json
metric_save_path=../results/${backbone_model_name}_${model_name}_test_metrics.csv
each_sample_metric_save_path=../results/${backbone_model_name}_${model_name}_test_each_sample_metrics.pkl
echo ${input_pred_result_path}
echo ${metric_save_path}
echo ${each_sample_metric_save_path}
CUDA_VISIBLE_DEVICES=2 python -u ../evaluate_metrics.py \
  --backbone_model_name=${backbone_model_name} \
  --input_pred_result_path=${input_pred_result_path} \
  --test_path=${test_path} \
  --metric_save_path=${metric_save_path} \
  --each_sample_metric_save_path=${each_sample_metric_save_path}