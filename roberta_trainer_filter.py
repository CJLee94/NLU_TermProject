from datasets import load_dataset, load_metric
from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, get_scheduler
import  torch
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

def roberta_trainer(dataset_type="mnli",threshold=0.99):
    # load the dataset and metric
    num_labels = 3
    if dataset_type=="qnli":
        num_labels=2

    dataset = load_dataset("glue", dataset_type)
    metric = load_metric("glue", dataset_type)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    # define a pretrain method
    def preprocess_function(examples):
        if dataset_type=="mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
        elif dataset_type=="rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        elif dataset_type=="qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True)

    # preprocess the data
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    ## data filtering
    aum_dir = "/scratch/sz2257/{}-aum".format('roberta')

    with open(os.path.join(aum_dir, "{}-{}_aum_0512.json".format('roberta', 'mnli')), "r") as f:
        aum_filter = json.load(f)
    filtered_list = aum_filter[str(threshold)]['intersection']

    train_set=encoded_dataset["train"]
    remain_data = list(set(range(len(train_set))) - set(filtered_list))
    train_set_filtered = train_set.filter(lambda item: item["idx"] in remain_data)

    print(len(filtered_list),len(remain_data))
    print("train dataset size: {} / {}".format(len(train_set),len(train_set_filtered)))


    # set all the training parameter
    batch_size =32
    args = TrainingArguments(
        "roberta-{}-{:.2f}-train-filtered-0512".format(dataset_type,threshold),
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
    validation_key="validation_matched" if dataset_type=="mnli" else "validation"
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset to use", default="mnli",type = str)
    parser.add_argument("-t", "--threshold", help="aum threshold", default=0.99,type=float) # [0.01,0.1,0.5,0.9,0.99]

    args = parser.parse_args()

    print(args)

    roberta_trainer(dataset_type=args.dataset,threshold=args.threshold)

"""
python NLU_TermProject/albert_trainer_filter.py --threshold 0.01
"""