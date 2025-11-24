
# bart-base-chinese, mengzi-t5-base, mt5base, Randeng-T5-784M
# vanilla, label_cond
model_name=vanilla
backbone_model_name=Randeng-T5-784M
test_path=/home/leyuquan/projects/LLMs/CVG/datasets/c3vg_dataset/CJO_test.json
test_pred_result_path=/home/leyuquan/projects/LLMs/CVG/criminal_cvg_plm/output/test_pred_results/${backbone_model_name}_${model_name}_test_result_200.json
metric_save_path=../results/${backbone_model_name}_${model_name}_test_macro_metrics.csv
each_label_metric_save_path=../results/${backbone_model_name}_${model_name}_test_each_label_metrics.csv
echo ${input_pred_result_path}
echo ${metric_save_path}
echo ${each_label_metric_save_path}
CUDA_VISIBLE_DEVICES=2 python -u ../evaluate_metrics_macro.py \
  --backbone_model_name=${backbone_model_name} \
  --test_path=${test_path} \
  --test_pred_result_path=${test_pred_result_path} \
  --metric_save_path=${metric_save_path} \
  --each_label_metric_save_path=${each_label_metric_save_path}
