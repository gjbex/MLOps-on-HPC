#!/usr/bin/env bash

set -euo pipefail

dvc stage add --name split_data --force \
    --deps data/data.csv \
    --deps src/split_data.py --deps src/utils.py \
    --params split_data.test_size,split_data.random_state \
    --outs data/split_data/train.csv \
    --outs data/split_data/test.csv \
    python src/split_data.py \
        --data data/data.csv \
        --params params.yaml \
        --output data/split_data/
git commit -m 'Add split_ddata stage'

dvc stage add --name train_preprocessor --force \
    --deps data/split_data/train.csv \
    --deps src/train_preprocessor.py --deps src/utils.py \
    --outs ./preprocessor.pkl \
    python src/train_preprocessor.py \
        --data data/split_data/train.csv \
        --output preprocessor.pkl
git commit -m 'Add train_preprocessor stage'

dvc stage add --name preprocess_train --force \
    --deps data/split_data/train.csv \
    --deps preprocessor.pkl \
    --deps src/preprocess.py --deps src/utils.py \
    --outs data/preprocessed/train.csv \
    python src/preprocess.py \
        --data data/split_data/train.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/train.csv
git commit -m 'Add preprocess_train stage'

dvc stage add --name preprocess_test --force \
    --deps data/split_data/test.csv \
    --deps preprocessor.pkl \
    --deps src/preprocess.py --deps src/utils.py \
    --outs data/preprocessed/test.csv \
    python src/preprocess.py \
        --data data/split_data/test.csv \
        --preprocessor preprocessor.pkl \
        --output data/preprocessed/test.csv
git commit -m 'Add preprocess_test stage'

dvc stage add --name train_model --force \
    --deps data/preprocessed/train.csv \
    --deps src/train_model.py --deps src/utils.py \
    --params train_model.penalty,train_model.C,train_model.solver \
    --outs model.pkl \
    python src/train_model.py \
        --data data/preprocessed/train.csv \
        --params params.yaml \
        --output model.pkl
git commit -m 'Add train_model stage'

dvc stage add --name compute_metrics_train --force \
    --deps data/preprocessed/train.csv \
    --deps model.pkl \
    --deps src/compute_metrics.py --deps src/utils.py \
    --metrics metrics/train.yaml \
    python src/compute_metrics.py \
        --data data/preprocessed/train.csv \
        --model model.pkl \
        --output metrics/train.yaml
git commit -m 'Add compute_metrics_train stage'

dvc stage add --name compute_metrics_test --force \
    --deps data/preprocessed/test.csv \
    --deps model.pkl \
    --deps src/compute_metrics.py --deps src/utils.py \
    --metrics metrics/test.yaml \
    python src/compute_metrics.py \
        --data data/preprocessed/test.csv \
        --model model.pkl \
        --output metrics/test.yaml
git commit -m 'Add compute_metrics_test stage'

