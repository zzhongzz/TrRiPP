import torch
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
import pandas as pd
import pprint
from pathlib import Path
from Bio import SeqIO
from data_loader import FasDataset, collate_fas
from models import LSTMNormBert
from train import config
from utils import RIPPS_CLASSES
from FocalLoss import FocalLoss, FocalWithLogitsLoss
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
num_to_ripp = {i: r for i, r in enumerate(RIPPS_CLASSES)}


def find_fasta_in_dir(dir_path):
    formats = ['fas', 'fa', 'faa', 'fasta']
    fas_paths = []
    for p in Path(dir_path).iterdir():
        if p.is_dir():
            continue
        if p.suffix.lower().replace(".", "") in formats:
            fas_paths.append(str(p))

    return fas_paths


class Prediction(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.config = conf
        self.lstm_bert = LSTMNormBert(conf)
        self.criterion = FocalWithLogitsLoss(gamma=self.config['focal_loss_gamma'])
        self.criterion_binary = FocalLoss(gamma=self.config['focal_loss_gamma'])

    def forward(self, input_ids, attention_mask):
        output, score, _ = self.lstm_bert(input_ids, attention_mask)
        return output, score


def predict(model, fasta_file, batch_size, length):
    if length > config["max_seq_length"]:
        length = config["max_seq_length"]
    model.eval()
    logging.info('loading fasta: {}'.format(fasta_file))
    fas = FasDataset(fasta_file, max_seq_length=length)
    dataloader = data.DataLoader(fas, batch_size=batch_size, shuffle=False, collate_fn=collate_fas,
                                 num_workers=24)
    num_seq = len(fas)
    if num_seq == 0:
        logging.warning(
            "no sequence to predict. Your fasta file might be empty or doesn't have the correct fast format.")
        return None, None, num_seq

    logging.info(f"{num_seq} sequences to predict")

    with torch.no_grad():
        all_predicts, all_prediction_scores, all_binary_scores = [], [], []
        all_ids, all_desc, all_seqs = [], [], []
        # can't use dict.from because: https://stackoverflow.com/questions/3000468/unwanted-behaviour-from-dict-fromkeys
        _ripps = {r: [] for r in RIPPS_CLASSES}

        for batch in dataloader:
            inputs, attention_mask, records = batch
            inputs = inputs.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            output, _ = model(inputs, attention_mask)

            predicted = torch.argmax(output, dim=1)
            score = F.softmax(output, dim=1)
            binary_score = 1 - score[:, 0]
            score, _ = torch.max(score, dim=1)

            all_predicts.append(predicted)
            all_prediction_scores.append(score)
            all_binary_scores.append(binary_score)
            all_seqs += [str(r.seq) for r in records]
            all_ids += [r.id for r in records]
            all_desc += [r.description for r in records]

            for i, p in enumerate(predicted):
                _class = p.item()
                ripp_class = num_to_ripp.get(_class)
                _ripps[ripp_class].append(records[i])

        all_predicts = torch.cat(all_predicts).cpu().numpy()
        all_prediction_scores = torch.cat(all_prediction_scores).cpu().numpy()
        all_binary_scores = torch.cat(all_binary_scores).cpu().numpy()

        df = pd.DataFrame({
            'filename': Path(fasta_file).stem,
            'id': all_ids,
            'description': all_desc,
            'sequence': all_seqs,
            'prediction': all_predicts,
            'class': [num_to_ripp.get(p) for p in all_predicts],
            'score': all_prediction_scores,
            'binary_score': all_binary_scores
        })

        logging.info("finished predicting fasta: {}".format(fasta_file))
        return df, _ripps, num_seq


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="predict RiPPs precursors from fasta files with protein/peptide sequences"
    )
    argparser.add_argument("-i", "--input", help="the fasta file or a directory of fasta files to predict",
                           required=True)
    argparser.add_argument("-o", "--output", help="the output file to collect the result", required=True)
    argparser.add_argument("-f", "--fasta", action="store_true",
                           help="write predicted RiPPs to fasta files (each class to one file, in the same directory as output)")
    argparser.add_argument("-w", "--weight", help="the weight file of the model",
                           default="./weight/best.ckpt")
    argparser.add_argument("-b", "--batch_size", default=512, type=int,
                           help="the batch size to perform prediction, larger batch size requires more memory")
    argparser.add_argument("-l", "--length", default=150, type=int,
                           help="max length of input sequence. Sequences above this length will be discarded")
    argparser.add_argument("-d", "--device",
                           help="specify whether using cpu or gpu to perform the prediction (gpu and cuda are the same)",
                           choices=["auto", "cpu", "gpu", "cuda"], default="auto")
    argparser.add_argument("--append", action="store_true", help="append results if they already exist")
    args = argparser.parse_args()

    if Path(args.input).is_dir():
        fasta_files = find_fasta_in_dir(args.input)
        if len(fasta_files) == 0:
            raise Exception("no fasta file found in {}".format(args.input))
    else:
        fasta_files = [args.input]

    if Path(args.output).exists() and not args.append:
        raise Exception(
            f'{args.output} output file exists! Use --append if you want to append result in the existing file')
    if args.fasta and not args.append:
        for key in RIPPS_CLASSES:
            out_file = Path(args.output).parent.joinpath(key + ".fas")
            if out_file.exists():
                raise Exception(
                    f'{out_file} output file exists! Use --append if you want to append result in the existing file')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == "cpu":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    logging.info("loading model...")
    model = Prediction(config).load_from_checkpoint(args.weight).to(DEVICE)
    model.eval()

    statistics = dict.fromkeys(["num_predicted_seqs", *RIPPS_CLASSES], 0)

    logging.info(f"{len(fasta_files)} fasta files to predict")
    for fasta_file in fasta_files:
        df, ripps, num_seq = predict(model, fasta_file, args.batch_size, args.length)
        statistics["num_predicted_seqs"] += num_seq
        if ripps is not None:
            for key in RIPPS_CLASSES:
                statistics[key] += len(ripps[key])

            if Path(args.output).exists():
                df.to_csv(args.output, mode="a", header=False, sep='\t', index=False)
            else:
                df.to_csv(args.output, sep='\t', index=False)

            if args.fasta:
                out_dir = Path(args.output).parent
                for key in RIPPS_CLASSES:
                    out_file = out_dir.joinpath(key + ".fas")
                    if out_file.exists():
                        out_file_handle = open(out_file, mode="a")
                        SeqIO.write(ripps[key], out_file_handle, "fasta")
                    else:
                        SeqIO.write(ripps[key], out_file, "fasta")

    pprint.pprint(statistics)
