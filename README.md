# Caojun-Time-aware-RE
## DSD-RE
基于远程监督的文档级关系抽取模型

### Denoise
训练去噪模块
```
python3 train.py -c config/PreDenoising.config -g 0
```

计算得分
```
mkdir data/rank_result

python3 test.py -g 0 \
    --config config/PreDenoising.config \
    --test_file data/train_distant.json \
    --checkpoint checkpoint/PreDenoise/127.pkl \
    --result_score data/rank_result/train_distant_score.npy \
    --result_title data/rank_result/train_distant_title.json 
```

### Pretrain
DocRED数据集位置 /docred_data
预处理数据
```
python3 gen_data.py --model_type bert --model_name_or_path bert-base-cased --data_dir docred_data --output_dir prepro_data --max_seq_length 512
```

预训练
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --model_type bert --model_name_or_path ../CorefBERT_base  --train_prefix train --test_prefix dev --evaluate_during_training_epoch 5 --prepro_data_dir prepro_data --max_seq_length 512 --batch_size 32 --learning_rate 4e-5 --num_train_epochs 200 --save_name DocRED_CorefBERT_base
```

### Fine-tune
微调模型
```
bash scripts/roberta-large_distant.sh
```

## PL-DocRTA
基于提示学习的文档级关系的时间感知模型

训练模型
```
bash scripts/t5.sh
```
