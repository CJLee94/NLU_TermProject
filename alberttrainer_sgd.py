from datasets import load_dataset, load_metric
from transformers import AlbertForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from utils import ALBERTTrainer
import argparse
import torch

import os
import json


def albert_trainer(dataset_type="mnli", aum=True, flip=True):
    out_dir = "albert-{}-train-{}-{}".format(dataset_type, aum, flip)
    os.makedirs(out_dir, exist_ok=True)

    # load the dataset and metric
    num_labels = 3
    if dataset_type == "qnli":
        num_labels = 2

    # if dataset_type == "snli":
    #     dataset = load_dataset(dataset_type)
    #     metric = load_metric("squad_v2")

    dataset = load_dataset("glue", dataset_type)
    metric = load_metric("glue", dataset_type)

    # randomly flip the labels in dataset
    if flip:
    # if True:
        save_file = os.path.join(out_dir, "flip_index.json")
        if os.path.exists(save_file):
            with open(save_file, "r") as fp:
                flip_index = json.load(fp)
        else:
            flip_index = []
            trainset = dataset["train"]
            length = len(trainset)
            samples_per_class = int(length/num_labels)
            samples_in_class = [samples_per_class for i in range(num_labels)]
            flips_in_class = [samples_per_class/(num_labels+1) for i in range(num_labels)]
            for i in range(length):
                c = trainset[i]["label"]
                samples = samples_in_class[c]
                flips = flips_in_class[c]
                if samples > 0:
                    p = flips/samples
                else:
                    p = 1
                indicator = np.random.choice([0, 1], p=[1-p, p])
                if indicator:
                    flip_index.append(trainset[i]["idx"])
                    flips_in_class[c] -= 1
                samples_in_class[c] -= 1
            with open(save_file, "w") as fp:
                json.dump(flip_index, fp)


    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2', use_fast=True)

    # define a pretrain method

    if aum:
        flip_samples = np.random

    def preprocess_function(examples):
        if dataset_type == "mnli" or dataset_type == "snli":
            feature = tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
        elif dataset_type == "rte":
            feature = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        elif dataset_type == "qnli":
            feature = tokenizer(examples["question"], examples["sentence"], truncation=True)
        # feature["idx"] = examples["idx"]
        return feature

    def preprocess_function_train(examples):
        feature = preprocess_function(examples)
        # if True:
        if flip:
            for ii, idx in enumerate(examples["idx"]):
                if idx in flip_index:
                    examples["label"][ii] = num_labels
        return feature

    # preprocess the data
    # encoded_dataset = dataset.map(preprocess_function, batched=True)
    validation_key = "validation_matched" if dataset_type == "mnli" else "validation"
    trainset = dataset["train"]
    evalset = dataset[validation_key]
    encoded_trainset = trainset.map(preprocess_function_train, batched=True)
    encoded_evalset = evalset.map(preprocess_function, batched=True)

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels+1)

    # ckpt_path="/media/felicia/Data/albert-{}-train/checkpoint-15000/".format(dataset_type)
    # model = AlbertForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)

    # set all the training parameter
    batch_size = 32

    # Default: AdamW
    args = TrainingArguments(
        out_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=15,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )
    
    steps_per_epoch = int(len(dataset["train"])//batch_size+1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, epochs=15, steps_per_epoch=steps_per_epoch, pct_start = 0.01, div_factor=25, final_div_factor=10)

    # define a metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return metric.compute(predictions=predictions, references=labels)

    # initialize trainer
    trainer = ALBERTTrainer(
        aum=aum,
        model=model,
        args=args,
        train_dataset=encoded_trainset,
        eval_dataset=encoded_evalset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(opt, lr_scheduler)
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
    parser.add_argument("-d", "--dataset", help="dataset to use", type = str, default="mnli")
    parser.add_argument("-a", "--aum", help="whether to enable aum", action="store_true")
    parser.add_argument("-s", "--syn", help="whether to flip some of the labels", action="store_true")
    args = parser.parse_args()

    albert_trainer(dataset_type=args.dataset, aum=args.aum, flip=args.syn)
