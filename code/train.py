import os
from subprocess import check_output
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model, EnsembleModel
from sts.utils import set_seed, setdir, check_params, make_file_name

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(args.data_dir, args.model_dir, reset=False)
    
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args)
    if not args.voting_models:
        #argparser로 받아온 parameter을 terminal에 출력하여 확인할 수 있게 합니다
        check_params(args.model_name, args.batch_size, args.max_epoch, args.shuffle, args.learning_rate, args.seed)
    
        model = Model(args.model_name, args.learning_rate)
        model_name = args.model_name.replace('/','_')
    else:
        print('Start Ensemble Learning...')
        for voting_model_name in args.voting_models:
            check_params(voting_model_name, args.batch_size, args.max_epoch, args.shuffle, args.learning_rate, args.seed)
        model = VotingModel(args.voting_models, args.learning_rate)
        model_name = model.model_name
        print(model_name)
    
    # checkpoint 
    ckpt_dirpath = setdir(args.data_dir, 'ckpt', reset=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{model_name}_{epoch:02d}_{val_loss:.4f}',
                                                  save_top_k=1, dirpath=ckpt_dirpath, monitor='val_pearson', mode='min')
    
    # learning-rate 모니터링 callback함수
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # wandb 설정
    if args.wandb:
        wandb_logger = WandbLogger(project=model_name,
                                   save_dir = '../data/wandb_checkpoints')
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=args.max_epoch, log_every_n_steps=1, logger=wandb_logger,
                             precision=16,
                             callbacks=[lr_monitor, checkpoint_callback])
    else:
        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator='gpu', devices=1,
                             max_epochs=args.max_epoch, log_every_n_steps=1)
    

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    if args.save_model:
        file_name = make_file_name(model_name, format='pt', version=args.version)
        model_path = os.path.join(dirpath, file_name)
        torch.save(model, model_path)


if __name__ == '__main__':
    args = get_args(mode="train")
    main(args)