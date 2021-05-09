from datasets import load_dataset, load_metric
from transformers import AlbertForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from utils import ALBERTTrainer
import argparse

def albert_trainer(dataset_type="mnli", aum=False):
    # load the dataset and metric
    num_labels = 3
    if dataset_type == "qnli":
        num_labels = 2

    # if dataset_type == "snli":
    #     dataset = load_dataset(dataset_type)
    #     metric = load_metric("squad_v2")

    dataset = load_dataset("glue", dataset_type)
    metric = load_metric("glue", dataset_type)

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
        feature["idx"] = examples["idx"]
        return feature

    # preprocess the data
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)

    # ckpt_path="/media/felicia/Data/albert-{}-train/checkpoint-15000/".format(dataset_type)
    # model = AlbertForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)

    # set all the training parameter
    batch_size = 32

    # Default: AdamW
    args = TrainingArguments(
        "albert-{}-train".format(dataset_type),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        save_steps=5000,
        save_total_limit=10,
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
    trainer = ALBERTTrainer(
        aum=aum,
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
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
    parser.add_argument("-d", "--dataset", help="dataset to use", type = str, default="qnli")
    parser.add_argument("-a", "--aum", help="whether to enable aum", action="store_true")
    args = parser.parse_args()

    albert_trainer(dataset_type=args.dataset, aum = args.aum)
