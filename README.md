# Second Language Acquisition of Neural Language Models
Scripts for preprocessing and trained models are coming soon!
```
# E.g.: Pretrain a model on Japanese cc100
l1=ja
export NGPU=4; CUDA_VISIBLE_DEVICES="$gpu_num" torchrun --nproc_per_node=$NGPU train.py \
    --exp_name "$exp_name" \
    --data_path "$processed_dataset_path" \
    --lgs "$l1" --mlm_steps "$l1" \
    --batch_size 520 \
    --epoch_size "$(< {pretrain_train_dataset_path} wc -l)" \
    --validation_metrics _valid_"$l1"_mlm_ppl

# E.g.: Finetuning the model on Tatoeba 
export NGPU=1; CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=$NGPU train.py \
    --exp_name "$exp_name"  \
    --data_path data/all/tatoeba/finetune/"$l1"-en \
    --lgs en-"$l1" --mlm_steps en-"$l1",en \
    --batch_size 250 \
    --epoch_size "$(< {finetune_train_dataset_path} wc -l)"\
    --validation_metrics _valid_en_"$l1"_mlm_ppl \
    --random_seed $random_seed
```