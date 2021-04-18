from datasets import load_dataset, load_metric
from transformers import AlbertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
import numpy as np

def albert_trainer(dataset_type="mnli"):
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
    def preprocess_function(examples):
        if dataset_type == "mnli" or dataset_type == "snli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)
        elif dataset_type == "rte":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
        elif dataset_type == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True)

    # preprocess the data
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # load the model
    model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)

    # ckpt_path="/media/felicia/Data/albert-{}-train/checkpoint-15000/".format(dataset_type)
    # model = AlbertForSequenceClassification.from_pretrained(ckpt_path, num_labels=num_labels)

    # set all the training parameter
    batch_size = 32
    args = TrainingArguments(
        "albert-{}-train".format(dataset_type),
        evaluation_strategy="epoch",
        # learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        # weight_decay=0.01,
        save_steps=5000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )


    opt=torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        # weight_decay=1e-4,
        # dampening=0,
        nesterov=True
    )

    steps_per_epoch=int(len(dataset["train"])//batch_size+1)

    # lr_scheduler=torch.optim.lr_scheduler.LambdaLR(
    #     optimizer=opt,
    #     lr_lambda= [lambda epoch:epoch],
    #     last_epoch=-1,
    #     verbose=True
    # )


    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     opt,
    #     mode='max',
    #     verbose=True,
    # )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=0.1,
        # total_steps=5,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        pct_start= 6/(5*steps_per_epoch)  #0.02
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
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(opt,lr_scheduler)
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

    albert_trainer(dataset_type="qnli")

""""
albert-base-v2:
{'eval_loss': 0.7939354777336121, 'eval_accuracy': 0.5053999633900788, 
'eval_runtime': 29.1222, 'eval_samples_per_second': 187.589, 
'init_mem_cpu_alloc_delta': 1843462144, 'init_mem_gpu_alloc_delta': 46745088, '
init_mem_cpu_peaked_delta': 33431552, 'init_mem_gpu_peaked_delta': 0, 'eval_mem_cpu_alloc_delta': 18046976, 
'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 49152, 'eval_mem_gpu_peaked_delta': 57188864}

ckpt-15000:
{'eval_loss': 0.49163737893104553, 'eval_accuracy': 0.9106717920556471, 
'eval_runtime': 28.0782, 'eval_samples_per_second': 194.564, 'init_mem_cpu_alloc_delta': 1845587968, 
'init_mem_gpu_alloc_delta': 46745088, 'init_mem_cpu_peaked_delta': 51974144, 
'init_mem_gpu_peaked_delta': 0, 'eval_mem_cpu_alloc_delta': 19054592,
 'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 86016,
  'eval_mem_gpu_peaked_delta': 57188864}
"""
