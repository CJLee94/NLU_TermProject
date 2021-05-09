from datasets import load_dataset, load_metric
from transformers import AlbertForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from utils import ALBERTTrainer
import argparse
import torch

import os
import json


def encode_dataset(dataset, tokenizer, num_labels, flip_index=None, dataset_type='mnli'):
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

    return encoded_trainset, encoded_evalset


def filter_dataset(out_dir, dataset, metric, tokenizer, num_labels, flip_index, dataset_type, aum=True, flip_name='1'):
    # encode the training and testing set
    encoded_trainset, encoded_evalset = encode_dataset(dataset, tokenizer, num_labels, flip_index, dataset_type)

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels + 1)

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
        num_train_epochs=7,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    steps_per_epoch = int(len(dataset["train"]) // batch_size + 1)
    opt = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.9, nesterov=True)

    # TODO change the scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, epochs=7, steps_per_epoch=steps_per_epoch,
                                                       pct_start=0.01, div_factor=25, final_div_factor=10)

    # define a metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # initialize trainer
    trainer = ALBERTTrainer(
        flip_index=flip_index,
        aum=aum,
        end_epoch=7,
        filter=True,
        flip_name=flip_name,
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
    return result


def albert_trainer(dataset_type="mnli", aum=True, filter=True):
    out_dir = "albert-{}-train-{}-filter-{}".format(dataset_type, aum, filter)
    os.makedirs(out_dir, exist_ok=True)

    # load the dataset and metric
    num_labels = 3
    if dataset_type == "qnli":
        num_labels = 2

    # if dataset_type == "snli":
    # dataset = load_dataset(dataset_type)
    # metric = load_metric("squad_v2")

    dataset = load_dataset("glue", dataset_type)
    metric = load_metric("glue", dataset_type)

    # randomly flip the labels in dataset
    save_file_1 = os.path.join(out_dir, "flip_index_1.json")
    save_file_2 = os.path.join(out_dir, "flip_index_2.json")


    if os.path.exists(save_file_1) and os.path.exists(save_file_2):
        with open(save_file_1, "r") as fp:
            flip_index_1 = json.load(fp)
        with open(save_file_2, "r") as fp:
            flip_index_2 = json.load(fp)

    # here is the generation, instead of generation 1 split, it will generate 2 split for the final calculation
    else:
        flip_index_1 = []
        flip_index_2 = []
        trainset = dataset["train"]
        length = len(trainset)
        samples_in_class_1 = np.unique(np.array(trainset["label"]), return_counts=True)[1]
        samples_in_class_2 = np.unique(np.array(trainset["label"]), return_counts=True)[1]
        flips_in_class_1 = [int(samples_in_class_1[i] / (num_labels + 1)) for i in range(num_labels)]
        flips_in_class_2 = [int(samples_in_class_2[i] / (num_labels + 1)) for i in range(num_labels)]
        for i in range(length):
            c = trainset[i]["label"]
            if flips_in_class_1[c] > 0:
                p = flips_in_class_1[c] / samples_in_class_1[c]
            else:
                p = 0

            if flips_in_class_2[c] > 0:
                q = flips_in_class_2[c] / samples_in_class_2[c]
            else:
                q = 0
            indicator = np.random.choice([0, 1, 2], p=[1 - p - q, p, q])
            if indicator == 1:
                flip_index_1.append(trainset[i]["idx"])
                flips_in_class_1[c] -= 1
            if indicator == 2:
                flip_index_2.append(trainset[i]["idx"])
                flips_in_class_2[c] -= 1
            samples_in_class_1[c] -= 1
            samples_in_class_2[c] -= 1
        with open(save_file_1, "w") as fp:
            json.dump(flip_index_1, fp)
        with open(save_file_2, "w") as fp:
            json.dump(flip_index_2, fp)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2', use_fast=True)

    # filter two the dataset two times
    # I don't know what or why result need to be print
    result = filter_dataset(out_dir, dataset, metric, tokenizer, num_labels, flip_index_1, dataset_type, flip_name='1')
    print(result)

    result = filter_dataset(out_dir, dataset, metric, tokenizer, num_labels, flip_index_2, dataset_type, flip_name='2')
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

    albert_trainer(dataset_type=args.dataset, aum=args.aum, filter=True)
