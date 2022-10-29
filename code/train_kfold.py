import os
import torch
import wandb
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from args import get_args
from sts.dataloader import KfoldDataloader
from sts.model import Model, KfoldModel
from sts.utils import set_seed

# 일단 급한대로
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    results = []
    
    # model을 생성합니다.
    model = KfoldModel(args.model_name, args.learning_rate)
    model_name = args.model_name.replace('/','_')
    
    # wandb 설정
    if args.wandb:
        wandb.login()
        wandb_logger = WandbLogger(project=model_name, save_dir = '../data/wandb_checkpoints')
        
    for k in range(args.num_folds):
        dataloader = KfoldDataloader(args, k)
        if args.wandb:
            trainer = pl.Trainer(accelerator='gpu', devices=1,
                                max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger)
        else:
            # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
            trainer = pl.Trainer(accelerator='gpu', devices=1,
                                max_epochs=args.max_epoch, log_every_n_steps=1)
    
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        score = trainer.test(model=model, datamodule=dataloader,verbose=True)
        results.extend(score)
    
    result = [x['k_test_pearson'] for x in results]
    score = sum(result) / args.num_folds
    
    if args.wandb:
        wandb.log({"test_pearson": score})
    print("K fold Test pearson: ", score)

    # 학습이 완료된 모델을 저장합니다.
    if args.save_model:
        model_name += args.version
        torch.save(model, f'../data/saved_models/{model_name}.pt')


if __name__ == '__main__':
    args = get_args(mode="train")
    main(args)