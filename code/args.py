import os
import sys
import argparse


def get_args(mode="train"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', default=42, type=int)
    # 유지관리
    parser.add_argument('--version', default='', type=str, required=True)
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--save_model', default=True)
    parser.add_argument('--wandb', default=True)
    # Hyperparameter
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    # Path
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--data_dir', default= '../data')
    parser.add_argument('--model_dir', default= 'saved_models')
    parser.add_argument('--save_dir', default= 'submissions')
    # K-fold
    parser.add_argument('--num_folds', default=5, type=int)
    parser.add_argument('--train_ratio', default=0.8)
    # checkpoint loader
    parser.add_argument('--checkpoint_path', default = '')
    # dev 데이터를 포함하여 학습시킬지 결정합니다.
    parser.add_argument('--use_dev',
                        help="Use dev data to train model. Use when you're going to submit.",
                        default=False)
    # 특수문자를 제거할지 결정합니다.
    parser.add_argument('--clean', default=False)
    
    return parser.parse_args(sys.argv[1:])