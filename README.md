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
## Citation
```
@inproceedings{oba-etal-2023-second,
    title = "Second Language Acquisition of Neural Language Models",
    author = "Oba, Miyu  and
      Kuribayashi, Tatsuki  and
      Ouchi, Hiroki  and
      Watanabe, Taro",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.856",
    doi = "10.18653/v1/2023.findings-acl.856",
    pages = "13557--13572",
    abstract = "With the success of neural language models (LMs), their language acquisition has gained much attention. This work sheds light on the second language (L2) acquisition of LMs, while previous work has typically explored their first language (L1) acquisition. Specifically, we trained bilingual LMs with a scenario similar to human L2 acquisition and analyzed their cross-lingual transfer from linguistic perspectives. Our exploratory experiments demonstrated that the L1 pretraining accelerated their linguistic generalization in L2, and language transfer configurations (e.g., the L1 choice, and presence of parallel texts) substantially affected their generalizations. These clarify their (non-)human-like L2 acquisition in particular aspects.",
}
```
