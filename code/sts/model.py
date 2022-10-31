
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        # 사용할 모델을 호출합니다.
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
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=self.lr*0.001)
        return [optimizer], [lr_scheduler]
    
class KfoldModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("k_test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        
class HuberModel(Model):
    def __init__(self, model_name, lr):
        super().__init__(model_name=model_name, lr=lr)
        
        self.loss_func1 = torch.nn.L1Loss()
        self.loss_func2 = torch.nn.HuberLoss()        

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func1(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func1(logits, y.float()) + self.loss_func2(logits, y.float())
        self.log("val_loss", loss)

        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss