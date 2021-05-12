from datasets import load_dataset, load_metric
from transformers import RobertaForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, get_scheduler
import  torch
import numpy as np


def roberta_trainer(dataset_type="mnli"):
    # load the dataset and metric
    num_labels = 3
    if dataset_type=="qnli":
        num_labels=2

    # if dataset_type == "snli":
    #     dataset = load_dataset(dataset_type)
    #     metric = load_metric("squad_v2")

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

    # load the model

    # ckpt_path="/media/felicia/Data/roberta-{}-train/checkpoint-60000/".format(dataset_type)
    # model = RobertaForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

    # set all the training parameter
    batch_size =32
    args = TrainingArguments(
        "roberta-{}-train-baseline".format(dataset_type),
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

    # opt=torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.1,
    #     weight_decay=1e-4,
    #     nesterov=True
    # )

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


if __name__ == '__main__':
    roberta_trainer(dataset_type="qnli")

""""
roberta-base:mnli
{'eval_loss': 1.100448489189148, 'eval_accuracy': 0.3273560876209883, 'eval_runtime': 38.7856, 
'eval_samples_per_second': 253.058, 'init_mem_cpu_alloc_delta': 1499799552, 
'init_mem_gpu_alloc_delta': 499890176, 'init_mem_cpu_peaked_delta': 380596224, 'init_mem_gpu_peaked_delta': 0, 
'eval_mem_cpu_alloc_delta': 19730432, 'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 364544, 
'eval_mem_gpu_peaked_delta': 40836096}

ckpt-60000:
{'eval_loss': 0.5340268611907959, 'eval_accuracy': 0.8758023433520122, 'eval_runtime': 37.019, 
'eval_samples_per_second': 265.134, 'init_mem_cpu_alloc_delta': 1497882624, 'init_mem_gpu_alloc_delta': 499890176,
 'init_mem_cpu_peaked_delta': 380366848, 'init_mem_gpu_peaked_delta': 0, 'eval_mem_cpu_alloc_delta': 19365888, 
 'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 503808, 'eval_mem_gpu_peaked_delta': 40836096}


"""
