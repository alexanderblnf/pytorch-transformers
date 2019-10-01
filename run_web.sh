TASK_NAME=mantis_web
BATCH_SIZE=32
SEED=10

for RANDOM_SEED in 10 100 1000 10000 100000
do
  python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=$BATCH_SIZE \
    --per_gpu_train_batch_size=$BATCH_SIZE \
    --save_steps=150 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./tmp/$TASK_NAME/$RANDOM_SEED \
    --gpu_id 0 \
    --seed $RANDOM_SEED \
    --eval_all_checkpoints \
    --overwrite_output_dir \
    --overwrite_cache && sleep 5
done