# config can be customized
config = {
    # model hyperparameters
    'lstm_layer': 2,
    'lstm_hidden': 64,
    'num_layer': 4,
    'head': 4,
    'embedding_dim': 128,
    'dim_feedforward': 256,
    'cls_dim': 192,
    # other hyperparameters
    'dropout': 0.4,
    'bert_dropout': 0.4,
    'lr_warmup': 10000,
    'lr': 1e-3,
    'max_seq_length': 256,
    'binary_loss_weight': 0,
    'focal_loss_gamma': 0,
    # training
    'batch_size': 128,
    'epoch': 200,
    'patience': 50,
    # classes in training data
    'num_classes': 10
}
