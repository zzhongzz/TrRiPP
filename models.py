import torch
import torch.nn as nn
import transformers as T
from data_loader import tokenizer
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMNormPacked(nn.Module):
    def __init__(self, input_size, out_size, num_layer, hidden_size, dropout, cls_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True, bidirectional=True,
            num_layers=num_layer, dropout=dropout
        )
        self.layerNorm1 = nn.LayerNorm(3 * 2 * hidden_size)
        self.dense = nn.Linear(3 * 2 * hidden_size, cls_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm2 = nn.LayerNorm(cls_dim)
        self.cls = nn.Linear(cls_dim, out_size)

    def forward(self, input_emb, attention_mask):
        lengths = torch.count_nonzero(attention_mask, dim=1)
        lengths_cpu = torch.count_nonzero(attention_mask, dim=1).cpu()
        # pack sequence to save calculations on pads
        packed = pack_padded_sequence(input_emb, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        # pad packed sequence to -inf so that torch.max won't extract values on pads
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=-float("Inf"))

        # get LSTM output of last token
        last_index = lengths - 1
        last_index = last_index.view(-1, 1, 1)
        last_index = last_index.expand(-1, 1, output.shape[2])
        output_last = output.gather(dim=1, index=last_index).squeeze(dim=1)
        # max pool
        output_max, _ = torch.max(output, dim=1)
        # average pool
        output_clone = output.clone()
        output_clone[attention_mask == 0] = 0
        output_avg = output_clone.sum(dim=1) / lengths.unsqueeze(1)
        # concat last, avg, max
        output_cat = torch.cat([output_last, output_avg, output_max], dim=1)

        output_cat = self.layerNorm1(output_cat)
        output_cat = self.dropout(self.relu(self.dense(output_cat)))
        output_cat = self.layerNorm2(output_cat)
        output_cat = self.cls(output_cat)

        return output_cat


class LSTMNormBert(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.config = conf

        self.bert_config = T.models.bert.modeling_bert.BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=self.config['embedding_dim'],
            num_hidden_layers=self.config['num_layer'],
            num_attention_heads=self.config['head'],
            intermediate_size=self.config['dim_feedforward'],
            max_position_embeddings=self.config['max_seq_length'],
            attention_probs_dropout_prob=self.config['bert_dropout'],
            hidden_dropout_prob=self.config['bert_dropout'],
            pad_token_id=tokenizer.pad_token_id
        )

        self.bert = T.models.bert.modeling_bert.BertModel(self.bert_config)
        self.lstm = LSTMNormPacked(self.config['embedding_dim'], self.config['num_classes'], self.config['lstm_layer'],
                                   self.config['lstm_hidden'], self.config['dropout'], self.config['cls_dim'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state

        output = torch.squeeze(self.lstm(emb, attention_mask), dim=1)
        # score = self.sigmoid(output)

        return output, None, 0
