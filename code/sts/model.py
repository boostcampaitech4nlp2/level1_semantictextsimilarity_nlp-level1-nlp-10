
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, model_name=None, lr=1e-5, scheduler=False):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.scheduler = scheduler

        # 사용할 모델을 호출합니다.
        if model_name:
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_name, num_labels=1)
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = torch.nn.L1Loss()

    def forward(self, x):
        x = self.plm(x)['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
    
    
class KfoldModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("k_test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))


class VotingModel(Model):
    def __init__(self, models, lr):
        super().__init__(lr=lr)
        
        # 사용할 모델을 호출합니다.
        model_list = []
        self.model_name = 'ensemble'
        for model in models: 
            model_name = model.replace('/','_')
            self.model_name = f'{self.model_name}+{model_name}'
            m = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model, num_labels=1)
            model_list.append(m)
        self.models = torch.nn.ModuleList(model_list)
            
    def forward(self,x):
        outs = None
        for i, model in enumerate(self.models):
            out = model(x)['logits']
            if i == 0: outs = out
            else: outs += out
        return outs


class StackingModel(VotingModel):
    def __init__(self, models, lr):
        super().__init__(models=models,lr=lr)
        self.linear = torch.nn.Linear(len(models), 1)
            
    def forward(self,x):
        outs = None
        for i, model in enumerate(self.models):
            out = model(x)['logits']
            if i == 0: outs = out.squeeze()
            else: outs = torch.stack((outs, out.squeeze())).transpose(1,0)
        ret = self.linear(outs)
        return ret

       
class HuberModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        self.loss_func2 = torch.nn.HuberLoss()        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss
    