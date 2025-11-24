mkdir ./output

python generate.py \
  --gpu_id=0 \
  --model_name=View_Gen \
  --model_path=/home/hcq/legalexp/courtviewgen/C3VG/bart_court_view/output/beer_view_gen_5 \
  --lr=1e-4 \
  --max_len=500 \
  --batch_size=12
