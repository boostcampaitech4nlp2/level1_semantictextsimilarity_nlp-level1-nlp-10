import os
import torch
import pytorch_lightning as pl
import pandas as pd

from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model, VotingModel
from utils import set_seed, setdir, make_file_name, hard_voting

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(args.data_dir, args.save_dir, reset=False)
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args)
    
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, log_every_n_steps=1)
    
    
    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model_name = args.model_name.replace('/','_') + '_' + args.version
    if args.checkpoint_path:
        model = Model(args.model_name, args.learning_rate)
        model = model.load_from_checkpoint(args.checkpoint_path)
    elif args.voting_models:
        model = VotingModel(args.voting_models, args.learning_rate)
        model.load_state_dict(torch.load(f'../data/saved_models/{model.model_name}_{args.version}.pt'))
    else:
        model = Model(args.model_name, args.learning_rate)
        model.load_state_dict(torch.load(f'../data/saved_models/{model_name}.pt'))
    
    print(f'Load Model:{model_name}...')
    trainer.test(model=model, datamodule=dataloader) #dev pearson 확인
    
    print(f'Make predictions....')
    predictions = trainer.predict(model=model, datamodule=dataloader)
    print('DONE')
    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/sample_submission.csv')
    output['target'] = predictions
    file_name = make_file_name(model_name, version=args.version, format='csv')
    save_path = os.path.join(dirpath, file_name)
    output.to_csv(save_path, index=False)

if __name__ == '__main__':
    args = get_args(mode="test")
    if args.use_hard_voting:
        hard_voting('../data/submissions', args.voting_submissions, args.version)
    else:
        main(args)
