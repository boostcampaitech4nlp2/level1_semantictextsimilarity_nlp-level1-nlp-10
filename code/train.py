from args import parse_args
from sts.dataloader import Dataloader
from sts.model import Model
from sts.utils import set_seed

import torch
import pytorch_lightning as pl


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)
    
    #TODO: Set Wandb
    # wandb setting
    # wandb.login()
    # wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args))

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    if args.save_model:
        torch.save(model, f'{args.model_name}.pt')


if __name__ == '__main__':
    args = parse_args(mode="train")
    main(args)


    
