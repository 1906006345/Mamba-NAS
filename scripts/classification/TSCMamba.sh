export CUDA_VISIBLE_DEVICES=0

model_name=TSCMamba
for classification_seed in 2021 2022 2023
do
  # python -u run.py \
  #   --task_name classification \
  #   --is_training 1 \
  #   --root_path ./data/UEA/Heartbeat/ \
  #   --model_id Heartbeat \
  #   --model $model_name \
  #   --data UEA \
  #   --e_layers 3 \
  #   --batch_size 16 \
  #   --d_model 16 \
  #   --d_ff 32 \
  #   --top_k 1 \
  #   --des 'MambaMultiLayer' \
  #   --itr 1 \
  #   --learning_rate 0.001 \
  #   --train_epochs 30 \
  #   --patience 10 \
  #   --classification_protocol fair \
  #   --classification_val_ratio 0.2 \
  #   --classification_monitor macro_f1 \
  #   --classification_seed $classification_seed
  
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./data/UEA/PEMS-SF/ \
    --model_id PEMS-SF \
    --model $model_name \
    --data UEA \
    --e_layers 3 \
    --batch_size 16 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 3 \
    --des 'MambaMultiLayer' \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --patience 10 \
    --classification_protocol fair \
    --classification_val_ratio 0.2 \
    --classification_monitor macro_f1 \
    --classification_seed $classification_seed 

  # python -u run.py \
  #   --task_name classification \
  #   --is_training 1 \
  #   --root_path ./data/UEA/SpokenArabicDigits/ \
  #   --model_id SpokenArabicDigits \
  #   --model $model_name \
  #   --data UEA \
  #   --e_layers 2 \
  #   --batch_size 16 \
  #   --d_model 32 \
  #   --d_ff 32 \
  #   --top_k 2 \
  #   --des 'MambaMultiLayer' \
  #   --itr 1 \
  #   --learning_rate 0.001 \
  #   --train_epochs 30 \
  #   --patience 10 \
  #   --classification_protocol fair \
  #   --classification_val_ratio 0.2 \
  #   --classification_monitor macro_f1 \
  #   --classification_seed $classification_seed
done
