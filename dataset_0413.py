import os
import numpy as np
import json
from tqdm import tqdm

from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset

from transformers import BertTokenizer,BertForSequenceClassification


class NLIDataset(Dataset):
    def __init__(self, root='/media/felicia/Data/{data}', data="multinli", split="train", padding=True, truncation=True, max_length=217):
        super(NLIDataset, self).__init__()
        self.root = root.format(data=data)
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.jsonfile = "{}_1.0_{}.jsonl".format(data, split)  # change format if needed
        self.filename = os.path.join(self.root, self.jsonfile)

        self.num_labels = 2
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)
        self.LABEL_MAP = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
            # "hidden": 0
        }

        self.data = []
        self.sentences_a =[]
        self.sentences_b =[]
        self.labels = []
        self.features=defaultdict(list)

        self.load_data()

    def load_data(self):
        with open(self.filename) as f:
            for line in f:
                example = json.loads(line)  # dict
                self.data.append(example)
        self.parseAll()

    def parseAll(self):
        for i, text in enumerate(self.data):
            sent1 = text["sentence1"]
            sent2 = text["sentence2"]
            label = text["gold_label"]
            if label not in self.LABEL_MAP:
                continue

            self.sentences_a.append(sent1)
            self.sentences_b.append(sent2)
            self.labels.append(self.LABEL_MAP[label])

        self.convertToFeatures()

    def convertToFeatures(self):
        for idx in tqdm(range(len(self.labels))):
            tokens_a = self.tokenizer.tokenize(self.sentences_a[idx])
            tokens_b = self.tokenizer.tokenize(self.sentences_b[idx])

            self._truncate_seq_pair(tokens_a, tokens_b, self.max_length - 3)

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a)+2)+[1]* (len(tokens_b)+1)

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
            self.features["input_mask"].append(input_mask)
            self.features["segment_ids"].append(segment_ids)
            self.features["label_id"].append(self.labels[idx])


    def _truncate_seq_pair(self,tokens_a, tokens_b,max_length):
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
        input_ids=self.features["input_ids"][index]
        input_mask=self.features["input_mask"][index]
        segment_ids=self.features["segment_ids"][index]
        label_ids=self.features["label_id"][index]

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), torch.tensor(label_ids)

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    """
    data="multinli"/"snli"  
    split=
        "train","dev_matched","dev_mismatched"
        "train","test","dev" fo "snli"
    tokenized=True(default) or False
    padding= True or False(default)
    max_length=12 (default)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    BATCH = 5

    print("Data loading ...")

    train_data = NLIDataset(data="multinli", split="train",  padding=False)
    train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True)

    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    # model.to(device)
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)


    for idx, batch in tqdm(enumerate(train_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        print(input_ids.size())  # tensor: b * max_length
        print(input_mask.size())  # tensor:  b * max_length
        print(segment_ids.size())  # tensor:  b * max_length
        print(label_ids.size())  # tensor: b

        if idx > 1:
            break