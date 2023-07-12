# TrRiPP

TrRiPP is a deep learning model to predict RiPP precursors directly from peptide sequences.

## How to use

### Installation

It is recommended to install dependencies by conda:

    conda env create -f environment.yml

After installation, activate the conda environment:

    conda activate trripp

Or you can install the following dependencies manually:

* [PyTorch](https://pytorch.org/get-started/locally/) 1.6.0+
* [Transformers](https://huggingface.co/docs/transformers/installation) 4.1.1+
* [PyTorch Lightning](https://pytorchlightning.ai/) 1.1.0+
* [TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/pages/quickstart.html) 0.10.0+
* [Biopython](https://biopython.org/wiki/Download)
* [scikit-learn](https://scikit-learn.org/stable/install.html)
* [matplotlib](https://matplotlib.org/stable/)
* [NumPy](https://numpy.org/install/)
* [pandas](https://pandas.pydata.org/getting_started.html)


### Predict RiPPs precursors

You can use the following command to predict sequences in fasta format. It usually requires ~3 GB of GPU RAM and ~4 GB
of CPU RAM to run a prediction with `batch_size` of 512 and `length` of 200 on cuda device. It takes ~10 seconds to
predict 15k sequences on a server with NVIDIA T4 GPU.

    python predict.py --input <input> --output <output> --weight ./weight/best.ckpt --batch_size 512 --length 200 --device cuda --fasta

`--input`: the fasta file or a directory of fasta files to predict. Fasta files should contain amino acid sequences
instead of nuclear acid sequences.

`--output`: the output file to collect the result. Result will be a tabular file with 5 columns: filename, fasta ID,
sequence, prediction score, binary prediction score. Prediction score is the probability of the sequence to be this
class, while binary prediction score is the probability the sequence to be any classed of RiPPs.

`--fasta`: a flag indicates whether to write predicted RiPPs to fasta files (each class to one file, in the same
directory as output)

`--weight`: the weight file of the model

`--batch_size`: the batch size to perform prediction, large batch size requires more memory but run faster

`--length`: the max length of input sequence. Sequences above this length will be discarded.

`--device`: specify whether using cpu or gpu to perform the prediction. Choose from `auto`, `cpu`, `gpu`, or `cuda`
which is the same as `gpu`.

### Train new model

You can use the following command to train the model. It usually requires ~6 GB of GPU RAM and ~4 GB of CPU RAM to
train the model with default config and datasets on cuda device. It takes ~5 hours to fully train the model on a server
with NVIDIA T4 GPU.

    python train.py --input ./data --output ./result/ --name multiclass_classification --device cuda

`--input`: the directory containing training, validation and testing dataset path. The directory should contain 3
files `train.txt`, `val.txt`, `test.txt`. All files should be tabular separated files, 2nd column is peptide sequences,
3 column is class of RiPPs, other columns could be anything. If `test.txt` is not provided, `val.txt` will be used to
test the model.

`--output`: the directory collecting the training result. The model can be found in the subdirectory with `.ckpt`
suffix (e.g. `multiclass_classification/version_0/checkpoints/epoch=0.ckpt`).

`--name`: the name of the mode

`--device`: specify whether using cpu or gpu to perform the prediction. Choose from `auto`, `cpu`, `gpu`, or `cuda`
which is the same as `gpu`.

`--seed`: set global random seed.

You can install tensorboard if you are interested in visualizing the training process.

    conda install -c conda-forge tensorboard
    tensorboard --log_dir=<output_dir>

#### Train new model with different hyperparameters

The config in `train.py` could be modified for training with different hyperparameters. The default hyperparameters used
in the paper is:

```
config = {
    'lstm_layer': 2,
    'lstm_hidden': 64,
    'num_layer': 4,
    'head': 4,
    'embedding_dim': 128,
    'dim_feedforward': 256,
    'cls_dim': 192,
    'dropout': 0.4,
    'bert_dropout': 0.4,
    'lr_warmup': 10000,
    'lr': 1e-3,
    'max_seq_length': 256,
    'binary_loss_weight': 0,
    'focal_loss_gamma': 1,
    'batch_size': 128,
    'epoch': 200,
    'patience': 50,
    'num_classes': 10
    ...
}
```
