import os
import warnings
import torch
import transformers
import pytorch_lightning as pl
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import re
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle

        self.train_path = args.train_path
        self.dev_path = args.dev_path
        self.test_path = args.test_path
        self.predict_path = args.predict_path
        
        self.use_dev = args.use_dev
        self.train_ratio = args.train_ratio
        self.num_workers = None
        if args.clean == True or args.clean == 'True' or args.clean == 'true':
            self.clean = True
        else:
            self.clean = False

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        
        if self.num_workers is None:
            num_workers = os.cpu_count()
        
        if num_workers is None:
            warnings.warn("could not infer cpu count automatically, setting it to zero")
            num_workers = 0
        self.num_workers = num_workers

        # 제거할 특수문자 (',', ?, !, '.' 제외)
        self.punctuation = '[-=+#/\:^@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\'…》]'

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def remove_punctuation(self, dataframe):
        df = dataframe.copy()
        sentences = df[self.text_columns].values
        refined_sentences = np.array(list(map(lambda x: [re.sub(self.punctuation, '', x[0]), re.sub(self.punctuation, '', x[1])], sentences)))
        df[self.text_columns] = refined_sentences
        return df

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []

        # 특수문자를 제거해야 할 경우 제거합니다.
        if self.clean:
            data = self.remove_punctuation(data)

        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            if self.use_dev:
                # 학습 데이터와 검증 데이터셋을 호출합니다
                train_data = pd.read_csv(self.train_path)
                val_data = pd.read_csv(self.dev_path)
                
            else:
                total_data = pd.read_csv(self.train_path)
                
                train_data = total_data.sample(frac=self.train_ratio)
                val_data = total_data.drop(train_data.index)
                
            # 학습, 검증 데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용.
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
            
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)
            
            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers = self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           num_workers = self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           num_workers = self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset,
                                           batch_size=self.batch_size,
                                           num_workers = self.num_workers)
    
class KfoldDataloader(Dataloader):
    def __init__(self,args,k):
        super().__init__(args=args)
        self.k = k
        self.num_folds = args.num_folds
        self.split_seed = args.seed
        
    def setup(self, stage='fit'):
        if stage == 'fit':
            # TODO: use_dev 구현
            # train 데이터와 dev 데이터를 합쳐서 total data를 구성해야 함.
            # 데이터를 호출합니다
            if self.use_dev:
                total_data = pd.concat([pd.read_csv(self.train_path), pd.read_csv(self.dev_path)])
            else:
                total_data = pd.read_csv(self.train_path)
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터셋 num_splits 번 fold
            kf = KFold(n_splits=self.num_folds, shuffle=self.shuffle, random_state=self.split_seed)
            all_splits = [k for k in kf.split(total_dataset)]
            
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]
            
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            predict_data = pd.read_csv(self.predict_path)
            
            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])