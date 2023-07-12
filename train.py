import torch
import torch.utils.data as data
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import pytorch_lightning.callbacks as cb
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pathlib import Path
from data_loader import ClassificationDataset, collate_for_predict
from utils import plot_cm, plot_roc
from models import *
from config import config


class Classify(pl.LightningModule):
    def __init__(self, conf, train_dataset_path, val_dataset_path, test_dataset_path):
        super().__init__()
        self.config = conf
        self.lr = self.config['lr']

        self.train_set = ClassificationDataset(train_dataset_path, self.config['max_seq_length'])
        num_training_pairs = len(self.train_set)
        print('num of training pairs:', num_training_pairs)

        self.eval_set = ClassificationDataset(val_dataset_path, self.config['max_seq_length'])
        self.num_training_data = len(self.train_set) * self.config['epoch']
        self.test_set = ClassificationDataset(test_dataset_path, self.config['max_seq_length'])

        # weight = self.train_set.get_labels_weight()
        # binary_weight = self.train_set.get_labels_weight_binary()
        weight, binary_weight = None, None
        if self.config['focal_loss_gamma'] > 0:
            from FocalLoss import FocalLoss, FocalWithLogitsLoss
            self.criterion = FocalWithLogitsLoss(alpha=weight, gamma=self.config['focal_loss_gamma'])
            self.criterion_binary = FocalLoss(alpha=binary_weight, gamma=self.config['focal_loss_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(weight=weight)
            self.criterion_binary = nn.NLLLoss(weight=binary_weight)

        self.lstm_bert = LSTMNormBert(conf)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        output, score, lm_loss = self.lstm_bert(input_ids=input_ids, attention_mask=attention_mask, lm_labels=lm_labels)
        return output, score, lm_loss

    def train_dataloader(self):
        return data.DataLoader(self.train_set, batch_size=self.config['batch_size'], shuffle=True,
                               collate_fn=collate_for_predict, num_workers=24)

    def val_dataloader(self):
        return data.DataLoader(self.eval_set, batch_size=self.config['batch_size'], shuffle=False,
                               collate_fn=collate_for_predict, num_workers=24)

    def test_dataloader(self):
        return data.DataLoader(self.test_set, batch_size=self.config['batch_size'], shuffle=False,
                               collate_fn=collate_for_predict, num_workers=24)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8, weight_decay=0.01)
        # num_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, epochs=self.config['epoch'],
        #                                                 steps_per_epoch=num_batches)
        scheduler = T.optimization.get_linear_schedule_with_warmup(optimizer, self.config['lr_warmup'],
                                                                   self.num_training_data / self.config['batch_size'])
        return [optimizer], \
            [{'scheduler': scheduler, 'interval': 'step'}]  # called after each training step

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, clm_labels = batch
        output, _, lm_loss = self.forward(input_ids=input_ids, attention_mask=attention_mask, lm_labels=clm_labels)
        binary_log_prob, binary_labels = self.calc_binary_labels_and_log_prob(output, labels)
        loss = (1 - self.config['binary_loss_weight']) * self.criterion(output, labels) + self.config[
            'binary_loss_weight'] * self.criterion_binary(binary_log_prob, binary_labels) + lm_loss
        self.log('train_step_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss = 0
        for output in outputs:
            train_loss += output['loss']

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, _ = batch
        output, _, lm_loss = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        predict = torch.argmax(output, dim=1)
        binary_log_prob, binary_labels = self.calc_binary_labels_and_log_prob(output, labels)
        loss = (1 - self.config['binary_loss_weight']) * self.criterion(output, labels) + self.config[
            'binary_loss_weight'] * self.criterion_binary(binary_log_prob, binary_labels)
        return {'loss': loss, 'true_label': labels.type(torch.int), 'predict_label': predict}

    def validation_epoch_end(self, outputs):
        all_predicts = torch.cat([o['predict_label'] for o in outputs])
        all_labels = torch.cat([o['true_label'] for o in outputs])
        acc = torchmetrics.functional.classification.accuracy(all_predicts, all_labels, task='multiclass',
                                                              num_classes=config['num_classes'])
        weighted_f1 = torchmetrics.functional.f1_score(all_predicts, all_labels, average="weighted", task='multiclass',
                                                       num_classes=self.config['num_classes'])
        kappa = torchmetrics.functional.cohen_kappa(all_labels, all_predicts, task='multiclass',
                                                    num_classes=self.config['num_classes'])
        val_loss = 0
        for output in outputs:
            val_loss += output['loss']
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)
        self.log('val_weight_f1', weighted_f1)
        self.log('val_kappa', kappa)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels, _ = batch
        output, _, lm_loss = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        predict = torch.argmax(output, dim=1)
        score = F.softmax(output, dim=1)
        binary_score = 1 - score[:, 0]
        score, _ = torch.max(score, dim=1)
        binary_log_prob, binary_labels = self.calc_binary_labels_and_log_prob(output, labels)
        loss = (1 - self.config['binary_loss_weight']) * self.criterion(output, labels) + self.config[
            'binary_loss_weight'] * self.criterion_binary(binary_log_prob, binary_labels)
        return {'loss': loss, 'true_label': labels.type(torch.int), 'predict_label': predict, 'prediction_score': score,
                'binary_score': binary_score}

    def test_epoch_end(self, outputs):
        all_predicts = torch.cat([o['predict_label'] for o in outputs])
        all_prediction_scores = torch.cat([o['prediction_score'] for o in outputs])
        all_binary_scores = torch.cat([o['binary_score'] for o in outputs])
        all_labels = torch.cat([o['true_label'] for o in outputs])
        test_loss = torch.stack([o['loss'] for o in outputs]).sum()
        acc = torchmetrics.functional.classification.accuracy(all_predicts, all_labels, task='multiclass',
                                                              num_classes=config['num_classes'])
        weighted_f1 = torchmetrics.functional.f1_score(all_predicts, all_labels, average="weighted", task='multiclass',
                                                       num_classes=self.config['num_classes'])
        kappa = torchmetrics.functional.cohen_kappa(all_labels, all_predicts, task='multiclass',
                                                    num_classes=self.config['num_classes'])
        all_labels_np = all_labels.cpu().numpy()
        all_predicts_np = all_predicts.cpu().numpy()
        all_binary_scores_np = all_binary_scores.cpu().numpy()
        cmt_figure = plot_cm(all_predicts_np, all_labels_np, self.config['num_classes'] == 1, loss=test_loss.item())
        self.logger.experiment.add_figure('confusion_matrix', cmt_figure)
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_weight_f1', weighted_f1)
        self.log('test_kappa', kappa)

        all_labels_binary = torch.ones_like(all_labels)
        all_labels_binary[all_labels == 0] = 0
        all_predicts_binary = torch.ones_like(all_predicts)
        all_predicts_binary[all_predicts == 0] = 0
        acc_binary = torchmetrics.functional.accuracy(all_predicts_binary, all_labels_binary, task='binary')
        stat_scores = torchmetrics.functional.stat_scores(all_predicts_binary, all_labels_binary, task='binary',
                                                          average='macro', num_classes=2)
        tn, fn, tp, fp, _ = stat_scores
        all_labels_binary_np = all_labels_binary.cpu().numpy()
        roc_figure = plot_roc(all_labels_binary_np, all_binary_scores_np)
        self.logger.experiment.add_figure('ROC curve', roc_figure)
        self.log_dict({'test_acc_binary': acc_binary, 'true_positive_binary': tp, 'false_positive_binary': fp,
                       'true_negative_binary': tn, 'false_negative_binary': fn})

    @staticmethod
    def calc_binary_labels_and_log_prob(_output, _labels):
        prob = F.softmax(_output, dim=1)
        binary_prob = torch.stack([prob[:, 0], torch.sum(prob[:, 1:], dim=1)], dim=1)
        binary_log_prob = torch.log(binary_prob)
        binary_labels = torch.ones_like(_labels)
        binary_labels[_labels == 0] = 0
        return binary_log_prob, binary_labels


def trainer_wrapper(conf, save_path, gpu, name):
    logger = TensorBoardLogger(save_path, name)
    lr_logger = pl.callbacks.LearningRateMonitor()
    ckpt_path = Path(logger.log_dir).joinpath('checkpoints')
    ckpt_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_kappa", mode="max", save_last=True,
                                                       save_weights_only=True)
    early_stopping_callback = cb.EarlyStopping(monitor='val_kappa', patience=conf['patience'], mode="max", verbose=True)
    # swa_callback = cb.StochasticWeightAveraging()
    return pl.Trainer(logger=logger,
                      callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
                      max_epochs=conf['epoch'],
                      # limit_train_batches=10,
                      # limit_val_batches=10,
                      # fast_dev_run=10,
                      precision=16,
                      gpus=gpu)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="train the model. The configs such as batch size and epoch can be modified in the script")
    argparser.add_argument("-i", "--input", default="./data",
                           help="the directory containing training, validation and testing dataset path.")
    argparser.add_argument("-o", "--output", default="./result/",
                           help="the directory collecting the training logs and model")
    argparser.add_argument("-n", "--name", default="multiclass_classification",
                           help="the name of the model")
    argparser.add_argument("-d", "--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto",
                           help="specify whether using cpu or gpu to perform the prediction")
    argparser.add_argument("-s", "--seed", type=int,
                           help="set global random seed. If not set, use random random seed")
    args = argparser.parse_args()
    if args.seed:
        pl.seed_everything(args.seed, workers=True)
    train_dataset_path = Path(args.input).joinpath('train.txt')
    val_dataset_path = Path(args.input).joinpath('val.txt')
    test_dataset_path = Path(args.input).joinpath('test.txt')
    if not test_dataset_path.exists():
        print('test.txt not exist, use val.txt as test set')
        test_dataset_path = val_dataset_path

    if args.device == "auto":
        cuda = 1 if torch.cuda.is_available() else 0
    elif args.device == "cpu":
        cuda = 0
    else:
        cuda = 1
    trainer = trainer_wrapper(config, args.output, cuda, args.name)
    model = Classify(config, train_dataset_path, val_dataset_path, test_dataset_path)

    # lr_finder = trainer.tuner.lr_find(model)
    # new_lr = lr_finder.suggestion()
    # print("new learning rate", new_lr)
    # model.lr = new_lr

    trainer.fit(model)

    print('testing best model:')
    trainer.test(ckpt_path='best')
    print('testing last model:')
    trainer.test(ckpt_path=trainer.checkpoint_callback.last_model_path)
