#from dataset_new import NLIDataset

import torch
from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert import BertConfig, BertAdam
# import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter


import os
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from torchvision import transforms, utils

import pickle
import time

from pytorch_pretrained_bert import BertTokenizer

from dataset_new import MnliProcessor, convert_examples_to_features
from tqdm import tqdm, trange
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


batch_size=24
epoch = 3
vocab_size = 30522
writer = SummaryWriter('runs/NLP_AUM_First')

processor = MnliProcessor()
num_labels = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
label_list = processor.get_labels()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)

train_examples = processor.get_train_examples("./multinli_1.0")

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

model.to(device)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01}, 
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

num_train_optimization_steps = int(len(train_examples) / batch_size) * epoch

optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5, warmup=0.1, t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0

train_features = convert_examples_to_features(train_examples, label_list, 217, tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", batch_size)
logger.info("  Num steps = %d", num_train_optimization_steps)

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

for _ in trange(int(epoch), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

        if (step+1)%20 == 0:
            print(tr_loss/nb_tr_steps/nb_tr_examples)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join("save_model", "pytorch_model.bin")
torch.save(model_to_save.state_dict(), output_model_file)





#config = BertConfig(vocab_size)
#model = BertForSequenceClassification(config, n_classes)
#model.to("cuda")
#dataset = NLIDataset("./multinli_1.0", tokenized=False, padding=False, max_length=217)
#dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
#criterion = torch.nn.CrossEntropyLoss()
#optimizer = BertAdam(model.parameters(), lr = 0.001)
#aum = torch.zeros([epoch, len(dataset), n_classes]).cuda()
#
#for e in range(epoch):
#    print("Epoch {}:".format(e))
#    logits_store = torch.zeros([len(dataset), n_classes]).cuda()
#    running_loss = 0.0
#    for idx, batch in enumerate(dataloader):
#        import pdb
#        pdb.set_trace()
#        optimizer.zero_grad()
#        sample_idx, text_in, labels = batch
#        logits = model(text_in.cuda())
#
#        logits_store[sample_idx] = logits
#
#        loss = criterion(logits, labels.cuda())
#        loss.backward()
#        optimizer.step()
#        running_loss+=loss.item()
#        if idx % 100 == 99:
#            print("Epoch {0}, Iter {1}:{2}".format(e, idx, running_loss/100))
#            running_loss = 0.0
#    logits_topk, logits_topk_ind = torch.topk(logits_store, 2, 1)
#    aum[e] = logits_store - logits_topk[:, 0][:, None]
#    aum[e, range(aum.shape[1]), logits_topk_ind[:,0]] = logits_store[range(aum.shape[1]), logits_topk_ind[:, 0]] - logits_topk[:,1]
##     aum[e, logits_topk_ind[0,1], 1] = logits_store[logits_topk_ind[0,1], 1] - logits_topk[1, 1]
#
#    attention_index = dataset.random_flip#Need the attention index to point out the flipped sample
#
#    for ind in attention_index:
#        writer.add_scalar("Sentence {}".format(ind), aum[e, ind, dataset[int(ind)][2]], e)
#    torch.save({"aum": aum, "flipped_sample":dataset.random_flip, "shift_label":dataset.random_shift}, "AUM_2.pth")
