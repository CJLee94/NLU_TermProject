import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import matplotlib.pyplot as plt

from dataset_0412 import NLIDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default="multinli")
parser.add_argument('--dataset_split', type=str,default="dev_matched")

args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()

sentset = NLIDataset(data=args.dataset_type, split=args.dataset_split, tokenized=True,padding=False, truncation=False)

BATCH = 1
contentloader = torch.utils.data.DataLoader(sentset, batch_size=BATCH, shuffle=False, num_workers=0)

sent1_list=[]
sent2_list=[]
label_list=[]

for idx, data in tqdm(enumerate(contentloader)):
    _, sent1, sent2, label = data
    sent1_list.append(sent1.numpy()[0])
    sent2_list.append(sent2.numpy()[0])
    label_list.append(label.numpy())

sent1_length=[len(x) for x in sent1_list]
sent2_length=[len(x) for x in sent2_list]


plt.grid()
plt.hist(sent1_length,30,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Length of Sentence',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('length of Sentence1 in {} {} Dataset'.format(args.dataset_type,args.dataset_split),fontsize=16,fontweight='bold')
plt.xticks()
plt.yticks()
# plt.show()
plt.savefig("vis/{}_{}_sentence1_hist.png".format(args.dataset_type,args.dataset_split))
plt.close()

plt.grid()
plt.hist(sent2_length,30,color='xkcd:turquoise', edgecolor='tab:gray')
plt.xlabel('Length of Sentence',fontsize=16)
plt.ylabel('Number of Users',fontsize=16)
plt.title('Length of Sentence2 in {} {} Dataset'.format(args.dataset_type,args.dataset_split),fontsize=16,fontweight='bold')
plt.xticks()
plt.yticks()
# plt.show()
plt.savefig("vis/{}_{}_sentence2_hist.png".format(args.dataset_type,args.dataset_split))
plt.close()


"""
ipython preprocess.py --dataset_split=train
"""