from datasets import load_dataset, load_metric
from transformers import AlbertForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AdamW
import torch
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

def albert_trainer(dataset_type="mnli", threshold=0.99):
    # load the dataset and metric
    num_labels = 3
    if dataset_type == "qnli":
        num_labels = 2


    dataset = load_dataset("glue", dataset_type)

    ## data filtering
    aum_dir = "/scratch/sz2257/{}-aum".format('albert')
    aum_1 = torch.load(os.path.join(aum_dir, "aum_1_6.pt"), map_location=torch.device('cpu')).detach().numpy()
    aum_2 = torch.load(os.path.join(aum_dir, "aum_2_6.pt"), map_location=torch.device('cpu')).detach().numpy()

    with open(os.path.join(aum_dir, 'flip_index_1.json')) as fp:
        flip_index_1 = json.load(fp)
    with open(os.path.join(aum_dir, 'flip_index_2.json')) as fp:
        flip_index_2 = json.load(fp)

    def filter_data(threshold=0.99):
        t1 = np.quantile(aum_1[flip_index_1], threshold)
        t2 = np.quantile(aum_2[flip_index_2], threshold)

        filter_list_1 = []
        filter_list_2 = []

        for i in tqdm(range(len(aum_1))):
            if aum_1[i] < t1:
                if i in flip_index_1:
                    pass
                else:
                    filter_list_1.append(i)

            if aum_2[i] < t2:
                if i in flip_index_2:
                    pass
                else:
                    filter_list_2.append(i)
            # if aum_1[i] < t1 and aum_2[i] < t2:
            # filter_list.append(i)

        # union_list = np.unique(filter_list) #
        union_list = list(set().union(filter_list_1, filter_list_2))
        intersection_list = list(set(filter_list_1).intersection(filter_list_2))

        return t1, t2, union_list, intersection_list

    t1, t2, union_list, intersection_list=filter_data(threshold=threshold)


    print("train dataset size: {} / {}".format(len(dataset),len(train_set)))

    metric = load_metric("glue", dataset_type)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2', use_fast=True)

    # define a pretrain method
    def preprocess_function(examples):
        if dataset_type == "mnli" or dataset_type == "snli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
        elif dataset_type == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        elif dataset_type == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True)

    # preprocess the data
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    train_set=encoded_dataset["train"]
    remain_data = list(set(range(len(train_set))) - set(union_list))
    train_set_filtered = train_set[remain_data]

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)

    # set all the training parameter
    batch_size = 32

    # Default: AdamW
    args = TrainingArguments(
        "albert-{}-{:.2f}-train-filtered".format(dataset_type,threshold),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )


    # define a metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return metric.compute(predictions=predictions, references=labels)

    # initialize trainer
    validation_key = "validation_matched" if dataset_type == "mnli" else "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set_filtered,
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # evaluate
    result = trainer.evaluate()

    # print the result
    print(result)


if __name__ == "__main__":
    """
    dataset_type: "mnli" , "rte"(Todo: "snli"), "qnli"
    P.S. "rte" is too small, glue does not include "snli"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset to use", default="mnli",type = str)
    parser.add_argument("-t", "--threshold", help="aum threshold", default=0.99,type=float) # [0.01,0.1,0.5,0.9,0.99]

    args = parser.parse_args()

    print(args)

    albert_trainer(dataset_type=args.dataset,threshold=args.threshold)

"""
python NLU_TermProject/albert_trainer_filter.py --threshold 0.01
"""