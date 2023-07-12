import torch
import torch.utils.data as data
from Bio import SeqIO
from transformers import BertTokenizer
from config import config

vocabFilePath = 'data/transformer_protein_vocab.txt'
tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False)


class ClassificationDataset(data.Dataset):
    def __init__(self, filename, max_seq_length=None):
        self.labels = []
        self.sequences = []
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        for line in lines:
            if len(line) != 0:
                _, sequence, _class = line.split("\t")[:3]
                labels = int(_class)
                seq = sequence.replace("*", "").upper()
                if max_seq_length is not None:
                    if 0 < len(seq) < max_seq_length:
                        self.labels.append(labels)
                        self.sequences.append(seq)
                else:
                    self.labels.append(labels)
                    self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.labels[index], self.sequences[index]

    # provides weights for the loss function, useful if training set is unbalanced
    def get_labels_weight(self):
        labels = self.labels
        print(torch.as_tensor([float(labels.count(i)) / float(len(labels)) for i in range(0, len(set(labels)))]))
        unique_labels = list(set(labels))
        unique_labels.sort()
        reversed_counts = [1 / labels.count(u) for u in unique_labels]
        s = sum(reversed_counts)

        weight = [item / s for item in reversed_counts]
        return torch.as_tensor(weight)

    # convert multi classes to binary, 0 as negative, 1 as positive
    def get_labels_weight_binary(self):
        labels = self.labels
        labels = [0 if x == 0 else 1 for x in labels]
        unique_labels = list(set(labels))
        unique_labels.sort()
        reversed_counts = [1 / labels.count(u) for u in unique_labels]
        s = sum(reversed_counts)

        weight = [item / s for item in reversed_counts]
        return torch.as_tensor(weight)


def collate_for_predict(batch):
    labels = [s[0] for s in batch]
    sequences_w_spaces = [" ".join(s[1]) for s in batch]
    encoded_sequences = tokenizer(sequences_w_spaces,
                                  add_special_tokens=True,
                                  return_token_type_ids=False,
                                  truncation='longest_first',
                                  padding=True,
                                  max_length=config['max_seq_length'],
                                  return_tensors='pt')

    input_ids = encoded_sequences.input_ids
    attention_mask = encoded_sequences.attention_mask
    labels = torch.as_tensor(labels).type_as(input_ids)
    # create labels for causal language model
    clm_labels = input_ids.clone()
    clm_labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels, clm_labels


class FasDataset(data.Dataset):
    def __init__(self, fas_filename, max_seq_length: int = None):
        if max_seq_length is not None:
            self.records = [r for r in SeqIO.parse(fas_filename, "fasta") if 0 < len(r.seq) <= max_seq_length]
        else:
            self.records = list(SeqIO.parse(fas_filename, "fasta"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        seq = str(self.records[index].seq).replace("*", "").upper()
        return seq, self.records[index]


def collate_fas(batch):
    sequences_w_spaces = [" ".join(s[0]) for s in batch]
    encoded_sequences = tokenizer(sequences_w_spaces,
                                  add_special_tokens=True,
                                  return_token_type_ids=False,
                                  truncation='longest_first',
                                  padding=True,
                                  max_length=config['max_seq_length'],
                                  return_tensors='pt')
    records = [s[1] for s in batch]
    input_ids = encoded_sequences.input_ids
    attention_mask = encoded_sequences.attention_mask

    return input_ids, attention_mask, records
