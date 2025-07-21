#!/usr/bin/env bash

dvc stage add --name split_data --force \
    --deps data/data.csv \
    --deps split_data.py --deps utils.py \
    --params split_data.test_size,split_data.random_state \
    --outs data/split_data/train.csv \
    --outs data/split_data/test.csv \
    python ./split_data.py \
        --data data/data.csv \
        --params params.yaml \
        --output data/split_data/

dvc stage add --name train_preprocessor --force \
    --deps data/split_data/train.csv \
    --deps train_preprocessor.py --deps utils.py \
    --outs ./preprocessor.pkl \
    python ./train_preprocessor.py \
        --data data/split_data/train.csv \
        --output preprocessor.pkl

dvc stage add --name preprocess_train --force \
    --deps data/split_data/train.csv \
    --deps preprocessor.pkl \
    --deps preprocess.py --deps utils.py \
    --outs data/preprocessed/train.csv \
    python ./preprocess.py \
        --data data/split_data/train.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/train.csv

dvc stage add --name preprocess_test --force \
    --deps data/split_data/test.csv \
    --deps preprocessor.pkl \
    --deps preprocess.py --deps utils.py \
    --outs data/preprocessed/test.csv \
    python ./preprocess.py \
        --data data/split_data/test.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/test.csv

dvc stage add --name train_model --force \
    --deps data/preprocessed/train.csv \
    --deps train_model.py --deps utils.py \
    --params train_model.penalty,train_model.C,train_model.solver \
    --outs model.pkl \
    python ./train_model.py \
        --data data/preprocessed/train.csv \
        --params params.yaml \
        --output model.pkl

dvc stage add --name compute_metrics_train --force \
    --deps data/preprocessed/train.csv \
    --deps model.pkl \
    --deps compute_metrics.py --deps utils.py \
    --metrics metrics/train.yaml \
    python ./compute_metrics.py \
        --data data/preprocessed/train.csv \
        --model model.pkl \
        --output metrics/train.yaml

dvc stage add --name compute_metrics_test --force \
    --deps data/preprocessed/test.csv \
    --deps model.pkl \
    --deps compute_metrics.py --deps utils.py \
    --metrics metrics/test.yaml \
    python ./compute_metrics.py \
        --data data/preprocessed/test.csv \
        --model model.pkl \
        --output metrics/test.yaml

