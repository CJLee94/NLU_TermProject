import os
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pickle
import time

from pytorch_pretrained_bert import BertTokenizer


class MultiNLIDataset(Dataset):
	def __init__(self,root='/media/felicia/Data/{data}', data='multinli',matched=True,tokenized=True,max_length=12):
		super(MultiNLIDataset,self).__init__()
		self.root=root.format(data=data)
		self.matched=matched
		self.tokenized=tokenized
		self.max_length=max_length
		self.jsonfile="multinli_1.0_dev_matched.jsonl" if self.matched else "multinli_1.0_dev_mismatched.jsonl"
		self.filename=os.path.join(self.root,self.jsonfile)

		
		self.num_labels = 2
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.LABEL_MAP = {
                "entailment": 0,
                "neutral": 1,
                "contradiction": 2,
                "hidden": 0
            }

		self.data=[]
		self.sentences=[]
		self.labels=[]

		self.load_data()

	def load_data(self):
		with open(self.filename) as f:
			for line in f:
				example=json.loads(line) # dict
				self.data.append(example)
		if self.tokenized:
			self.tokenize()

    
	def tokenize(self):
		for i, text in enumerate(self.data):
			sent=text["sentence1"]
			label=text["gold_label"]
			if label not in self.LABEL_MAP:
				continue

			tokenized_text = self.tokenizer.tokenize(sent)
			indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
			if len(indexed_tokens)<self.max_length:
				indexed_tokens+=[0]*(self.max_length-len(indexed_tokens))
			else:
				indexed_tokens=indexed_tokens[:self.max_length]
			indexed_tokens=np.array(indexed_tokens)

			self.sentences.append(indexed_tokens)
			self.labels.append(self.LABEL_MAP[label])
    
	def __getitem__(self,index):
		sent = self.sentences[index]
		label = self.labels[index]

		return index,sent,label

	def __len__(self):
		return len(self.data)


if __name__ == "__main__":
	BATCH=1
	sentset=MultiNLIDataset(max_length=15)  ## @ acgan-pytorch
	contentloader=torch.utils.data.DataLoader(sentset,batch_size=BATCH,shuffle=True,num_workers=4)

	for idx,data in enumerate(contentloader):
		index,sent, label=data 
		print(index) # tensor: b
		print(sent) # tensor:  b * max_length
		print(label) # tensor: b

		if idx>10:
			break