import os
import numpy as np
import json
import csv
from tqdm import tqdm

from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer, AlbertTokenizer
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, \
    RobertaForSequenceClassification


class NLIDataset(Dataset):
    def __init__(self, root='/media/felicia/Data/{data}', dataset="multinli", split="train", type="Roberta",
                 tokenizer=None,max_length=217):
        super(NLIDataset, self).__init__()
        self.dataset=dataset
        self.split=split
        self.root = root.format(data=self.dataset)
        if self.dataset=="multinli" or self.dataset== "snli":
            self.file = "{}_1.0_{}.jsonl".format(self.dataset, self.split)  # change format if needed
        else:
            self.file="{}.tsv".format(split)
        self.filename = os.path.join(self.root, self.file)
        self.max_length = max_length

        self.num_labels = 3
        self.LABEL_MAP = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
            "not_entailment": 1 # "qnli"
        }
        self.type = type
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", max_length=self.max_length)
        else:
            self.tokenizer = tokenizer

        self.data = []
        self.sentences_a = []
        self.sentences_b = []
        self.labels = []
        self.features = defaultdict(list)

        self.load_data()

    def load_data(self):
        with open(self.filename) as f:
            if self.dataset == "multinli" or self.dataset == "snli":
                for line in f:
                    example = json.loads(line)  # dict
                    self.data.append(example)
            else:
                tsv=csv.DictReader(f,delimiter="\t")
                for row in tsv:
                    self.data.append(row)
        self.parseAll()

    def parseAll(self):
        for i, text in enumerate(self.data):
            if self.dataset=="multinli" or self.dataset=="snli":
                sent1 = text["sentence1"]
                sent2 = text["sentence2"]
                label = text["gold_label"]
            else:
                sent1 = text["question"] # question
                sent2 = text["sentence"] # sentence
                label = "entailment" if self.split=="test" else text["label"]

            if label not in self.LABEL_MAP:
                continue

            self.sentences_a.append(sent1)
            self.sentences_b.append(sent2)
            self.labels.append(self.LABEL_MAP[label])

        if self.type == "albert":
            self.albertConvertToFeatures()
        elif self.type == "roberta":
            self.robertaConvertToFeatures()

    def albertConvertToFeatures(self):
        for idx in tqdm(range(len(self.labels))):
            tokens_a = self.tokenizer.tokenize(self.sentences_a[idx])[:-1] # remove .
            tokens_b = self.tokenizer.tokenize(self.sentences_b[idx])[:-1]

            self._truncate_seq_pair(tokens_a, tokens_b, self.max_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_length
            assert len(input_mask) == self.max_length
            assert len(segment_ids) == self.max_length

            self.features["input_ids"].append(input_ids)
            self.features["attention_mask"].append(input_mask)
            self.features["token_type_ids"].append(segment_ids)
            self.features["label"].append(self.labels[idx])

    def robertaConvertToFeatures(self):
        for idx in tqdm(range(len(self.labels))):
            tokens_a = self.tokenizer.tokenize(self.sentences_a[idx])[:-1]
            tokens_b = self.tokenizer.tokenize(self.sentences_b[idx])[:-1]

            self._truncate_seq_pair(tokens_a, tokens_b, self.max_length - 3)

            tokens = ["<s>"] + tokens_a + ["</s>"] + tokens_b + ["</s>"]
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # input_ids=tokens

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.max_length
            assert len(input_mask) == self.max_length
            assert len(segment_ids) == self.max_length

            self.features["input_ids"].append(input_ids)
            self.features["attention_mask"].append(input_mask)
            self.features["token_type_ids"].append(segment_ids)
            self.features["label"].append(self.labels[idx])

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __getitem__(self, index):
        input_ids = self.features["input_ids"][index]
        input_mask = self.features["attention_mask"][index]
        segment_ids = self.features["token_type_ids"][index]
        label_ids = self.features["label"][index]

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(label_ids)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    """
    dataset="multinli"/"snli" /"qnli"
    split=
        "train","dev_matched","dev_mismatched"
        "train","test","dev" fo "snli"/"qnli"
    num_labels=3 for "multinli"/"snli"
                2 for "qnli"
                
    Change dataset folder to multinli"/"snli" /"qnli"
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("Number of GPU:",n_gpu)

    BATCH = 8

    print("Data loading ...")

    ## tokenizer
    # BTokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
    ATokenizer=AlbertTokenizer.from_pretrained("albert-base-v1")
    # RTokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    albert_data = NLIDataset(type="albert",tokenizer=ATokenizer,dataset="qnli", split="train", max_length=217)
    # roberta_data = NLIDataset(type="roberta", tokenizer=RTokenizer, dataset="snli", split="test", max_length=217)

    albert_dataloader = DataLoader(albert_data, batch_size=BATCH, shuffle=False)
    # roberta_dataloader = DataLoader(roberta_data, batch_size=BATCH, shuffle=True)

    ## model
    # model=BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v1",num_labels=2 ,return_dict=False)
    # model = RobertaForSequenceClassification.from_pretrained("roberta-base",num_labels=3,return_dict=False)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    print("Transfomer: Albert")
    # print("Transfomer: RoBERTa")

    for idx, batch in tqdm(enumerate(albert_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        print(batch)

        # print(input_ids.size())  # tensor: b * max_length
        # print(input_mask.size())  # tensor:  b * max_length
        # print(segment_ids.size())  # tensor:  b * max_length
        # print(label_ids.size())  # tensor: b


        loss, logits = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids,labels=label_ids) # Albert
        # loss, logits = model(input_ids, attention_mask=input_mask,labels=label_ids) # Roberta


        print(loss)

        break



"""
CUDA_LAUNCH_BLOCKING=1 ipython dataset_0413.py

"""