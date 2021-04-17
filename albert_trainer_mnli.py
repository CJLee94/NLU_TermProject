from datasets import load_dataset, load_metric
import transformers
from transformers import AlbertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np


def albert_trainer():
    # load the dataset and metric
    dataset = load_dataset("glue", 'mnli')
    metric = load_metric('glue', 'mnli')

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('albert-base-v2', use_fast=True)

    # define a pretrain method
    def preprocess_function(examples):
        return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

    # preprocess the data
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=3)

    # set all the training parameter
    batch_size = 16
    args = TrainingArguments(
        "test-glue",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    # define a metric function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        return metric.compute(predictions=predictions, references=labels)

    # initialize trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset['validation_matched'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # train
    trainer.train()

    # evaluate
    result = trainer.evaluate()

    # print the result
    print(result)


if __name__ == '__main__':
    albert_trainer()
