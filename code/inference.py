import os
import torch
import pytorch_lightning as pl
import pandas as pd

from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model
from sts.utils import set_seed, setdir

data_dir = '../data'
save_dirname = 'saved_models'

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(data_dir,save_dirname, reset=False)
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)
    
    model_name = args.model_name.replace('/','_')
    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = torch.load(f'../data/saved_models/{model_name}.pt')
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions
    
    save_path = os.path.join(dirpath, f'{model_name}.pt')
    output.to_csv(save_path, index=False)

if __name__ == '__main__':
    args = get_args(mode="test")
    main(args)
