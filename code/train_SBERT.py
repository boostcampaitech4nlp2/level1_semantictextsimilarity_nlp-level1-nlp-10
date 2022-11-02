import os
from subprocess import check_output
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from args import get_args
from sts.dataloader import Dataloader
from sts.model import Model
from sts.utils import set_seed, setdir, check_params, make_file_name
from sentence_transformers import SentenceTransformer, models
from datetime import datetime
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
from sentence_transformers import losses
import math
from datetime import datetime
import sys
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
from sentence_transformers.readers import InputExample
from sentence_transformers.util import batch_to_device

'''
모델 load
 - SentenceTransformer.load(input_path)
   - 그냥 저장할 때 사용한 경로를 그대로 사용하는 듯
'''
    



def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    dirpath = setdir(args.data_dir, args.model_dir, reset=False)
    batch_size = int(args.batch_size)
    epochs = int(args.max_epoch)
    
    #argparser로 받아온 parameter을 terminal에 출력하여 확인할 수 있게 합니다
    check_params(args.model_name, args.batch_size, args.max_epoch, args.shuffle, args.learning_rate, args.seed)
    
    # 1. 기존 dataloader를 사용하지 않고 dataloder를 만듭니다.
    #    기존 dataloader가 [SEP]토큰을 사이에 두고 두 문장을 연결해서 하나의 문장을 만들어주는 건데,
    #    SBERT에서는 두 문장을 각각 따로 입력해야 하기 때문에 기존 dataloader를 사용하지 않았습니다.
    train_df = pd.read_csv(args.train_path)
    dev_df = pd.read_csv(args.dev_path)
    train_examples = make_sts_input_example(train_df)
    dev_examples = make_sts_input_example(dev_df)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    dev_ㅇㅁㅅㅁloader = DataLoader(
        dev_examples,
        batch_size=batch_size
    )
    
    # 2. embedding 모델과 pooling 모델을 생성하고 SentenceTransformer로 연결합니다.
    model_name = 'klue/roberta-base'
    #model_name = 'klue/roberta-large'
    embedding_model = models.Transformer(
        model_name_or_path=model_name,
        max_seq_length=256,
        do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[embedding_model, pooling_model])
    
    # 3. 모델 저장 경로를 지정하고 model.fit()으로 학습합니다.
    #    losses.CosineSimilarityLoss() 함수를 사용하면 CosineSimilarityLoss을 사용하는 모델을 리턴합니다.
    #    그러나 train시에만 CosineSimilairtyLoss로 감싸서 사용하고,
    #    이후 inference 시에는 감싼 모델이 아니라 감싸지지 않은 모델을 사용하는 것 같습니다.
    root_path = '../data/sbert/'
    save_path = root_path + model_name.replace('/', '-') + '/' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_epochs' + str(epochs)

    train_loss = losses.CosineSimilarityLoss(model=model)
    warmup_steps = math.ceil(len(train_examples) * epochs / batch_size * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        #evaluator=dev_evaluator,
        epochs=epochs,
        #evaluation_steps=int(len(train_dataloader)*0.1),
        warmup_steps=warmup_steps,
        output_path=save_path
    )
    
    # 4. 여기에 dev.csv로 pearson 점수를 계산하는 부분이 필요합니다!
    
    
# input data를 InputExample 데이터들의 리스트로 만들어서 리턴합니다.
# InputExample은 StentenceTrnasformer로 입력되는 데이터 형식입니다.
def make_sts_input_example(dataset):
    input_examples = []
    for i, data in tqdm(dataset.iterrows(), desc='SBERT input_examples', total=len(dataset)):
        st1 = data['sentence_1']
        st2 = data['sentence_2']
        score = (data['label']) / 5.0
        input_examples.append(InputExample(texts=[st1, st2], label=score))
    return input_examples

# dev.csv로 pearson 점수를 계산하는 함수를 만들어보는 중입니다.
# 현재 밑에 load_test() 함수 안에서 사용되고 있습니다.
def evaluate(model, dataloader):
    model.eval()
    dataloader.collate_fn = model.smart_batching_collate
    device = model._target_device
    with torch.no_grad():
        embeddings
        for features, labels in dataloader:
            # features, model, labels를 같은 메모리에 올려놓기
            features = list(map(lambda batch: batch_to_device(batch, model._target_device), features))
            model.to(device)
            labels = labels.to(device)
            '''
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py
            https://github.com/UKPLab/sentence-transformers/blob/83eeb5a7b9b81d17a235d76e101cc2912ee1a30d/sentence_transformers/util.py#L300
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py
            https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py
            https://github.com/UKPLab/sentence-transformers/blob/83eeb5a7b9b81d17a235d76e101cc2912ee1a30d/sentence_transformers/evaluation/SentenceEvaluator.py#L1
            https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/trainer/trainer.py
            
            EmbeddingSimilarityEvaluator가 어떻게 evaluate하는지 봐야 함.
            이에 따라 Models.encode()부분도 볼 필요가 있는 듯.
            Models.encode() 부분은 for문을 돌 필요 없이 한번에 알아서 처리하는 듯.
            '''
            pred = model(features, labels)
            
        
# 기존 코드에서 pearson 점수를 계산하는 부분입니다.
def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
    
# 모델을 불러온 뒤 dev.csv 파일을 사용하여 점수를 계산하는 함수입니다.
def load_test(args, model_path):
    root_path = 'data/sbert/'
    model = SentenceTransformer.load(model_path)
    
    dev_df = pd.read_csv(args.dev_path)
    dev_examples = make_sts_input_example(dev_df)
    
    dataloader = DataLoader(
        dev_examples,
    )
    evaluate(model, dataloader)
    
    
if __name__ == '__main__':
    args = get_args(mode="train")
    #main(args)
    #load_test(args, 'data/sbert/klue-roberta-base/2022-11-01_17-37-29_epochs1/')