import os
from subprocess import check_output
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model
from sts.utils import set_seed, setdir, check_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"
data_dir = '../data'
model_dirname = 'saved_models'

def main(args):
    global data_dir 
    global model_dir
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(data_dir, model_dirname, reset=False)
    
    #argparser로 받아온 parameter을 terminal에 출력하여 확인할 수 있게 합니다
    check_params(args.model_name, args.batch_size, args.max_epoch, args.shuffle, args.learning_rate, args.seed)
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args)
    model = Model(args.model_name, args.learning_rate)

    model_name = args.model_name.replace('/','_')
    # wandb 설정
    if args.wandb:
        wandb_logger = WandbLogger(project=model_name,
                                   save_dir = '../data/wandb_checkpoints')
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger)
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=args.max_epoch, log_every_n_steps=1)
    

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    # TODO: 중복 이름으로 덮어쓰기 되는 상황을 막으려면 랜덤 변수를 따로 이름에 설정해야 할 듯.
    if args.save_model:
        model_name += args.version
        model_path = os.path.join(dirpath, f'{model_name}.pt')
        torch.save(model, model_path)


if __name__ == '__main__':
    args = get_args(mode="train")
    main(args)


    
