import os
import numpy as np
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


class NLIDataset(Dataset):
	def __init__(self,root='/media/felicia/Data/{data}', data="multinli",split="train",tokenized=True,padding=False,max_length=12):
		super(NLIDataset,self).__init__()
		self.root=root.format(data=data)
		self.tokenized=tokenized
		self.padding=padding
		self.max_length=max_length
		self.jsonfile="{}_1.0_{}.jsonl".format(data,split) # change format if needed
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
		self.sentences={"sent1":[],"sent2":[]}
		self.labels=[]

		self.load_data()

	def load_data(self):
		with open(self.filename) as f:
			for line in f:
				example=json.loads(line) # dict
				self.data.append(example)
		self.parseAll()


	def tokenizeSent(self,sentence):

		tokenized_text = self.tokenizer.tokenize(sentence)
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

		if self.padding and len(indexed_tokens) < self.max_length:
			indexed_tokens += [0] * (self.max_length - len(indexed_tokens))
		else:
			indexed_tokens = indexed_tokens[:self.max_length]
		indexed_tokens = np.array(indexed_tokens)

		return indexed_tokens


	def parseAll(self):
		for i, text in tqdm(enumerate(self.data)):
			sent1=text["sentence1"]
			sent2=text["sentence2"]
			label=text["gold_label"]
			if label not in self.LABEL_MAP:
				continue

			if self.tokenized:
				sent1=self.tokenizeSent(sent1)
				sent2=self.tokenizeSent(sent2)

			self.sentences["sent1"].append(sent1)
			self.sentences["sent2"].append(sent2)
			self.labels.append(self.LABEL_MAP[label])
    
	def __getitem__(self,index):
		sent1 = self.sentences["sent1"][index]
		sent2 = self.sentences["sent2"][index]
		label = self.labels[index]

		return index,sent1,sent2,label

	def __len__(self):
		return len(self.data)


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

	sentset=NLIDataset(data='snli',split="test",tokenized=False,padding=False)

	BATCH=1
	contentloader=torch.utils.data.DataLoader(sentset,batch_size=BATCH,shuffle=True,num_workers=4)

	for idx,data in enumerate(contentloader):
		index,sent1,sent2, label=data
		print(index) # tensor: b
		print(sent1) # tensor:  b * max_length
		print(sent2) # tensor:  b * max_length
		print(label) # tensor: b

		if idx>10:
			break