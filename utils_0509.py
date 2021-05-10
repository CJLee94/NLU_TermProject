from typing import Optional, Dict, Union, Any
import inspect
from packaging import version
import os
from tqdm import tqdm

import torch
import torch.nn as nn

# from aum import AUMCalculator

from transformers.file_utils import is_apex_available
from transformers import Trainer
from transformers.utils import logging
import numpy as np
import json

logger = logging.get_logger(__name__)

#if is_sagemaker_mp_enabled():
#    from transformers.trainer_pt_utils import smp_forward_backward

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_apex_available():
    from apex import amp


class ALBERTTrainer(Trainer):
    def __init__(self, flip_index, aum=False, end_epoch=7, filter=False, flip_name='1', **kwargs):
        super().__init__(**kwargs)
        self.end_epoch = end_epoch
        # self.current_epoch = 0
        self.current_step=0
        self.flip_index = flip_index
        self.filter = filter
        self.filter_set="train"
        self.flip_name = flip_name
        if aum:
            self.aum = torch.zeros((len(self.train_dataset),))
            self.epoch_counting = torch.zeros((len(self.train_dataset),), dtype=torch.int8)
        #    self.aum = AUMCalculator(self.args.output_dir, compressed=False)
        else:
            self.aum = None

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids", "idx"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        dataset.set_format(type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"])

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)


        # if is_sagemaker_mp_enabled():
        # loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        # return loss_mb.reduce_mean().detach().to(self.args.device)
        if self.use_amp:
            with autocast():
                loss, outputs= self.compute_loss(model, inputs,return_outputs=True)
        else:
            loss, outputs = self.compute_loss(model, inputs,return_outputs=True)


        ## don't put compute aum in compute_loss, eval step also use compute_loss
        # if self.aum is not None:
        #     #records = self.aum.update(outputs["logits"], inputs["labels"], indices)
        #     self.compute_aum(indices, outputs, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # if "idx" in inputs:
        #     indices = inputs.pop("idx")
        indices = inputs.pop("idx")
        outputs = model(**inputs)


        if self.aum is not None:
            #records = self.aum.update(outputs["logits"], inputs["labels"], indices)
            self.compute_aum(indices, outputs, inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_aum(self, indices, outputs, inputs):
        # self.current_epoch += 1

        if self.filter_set=="train":

            self.aum = self.aum.to(indices.device)
            self.epoch_counting = self.epoch_counting.to(indices.device)
            outputs_c = outputs["logits"].clone()
            labels = inputs["labels"]
            assigned_logit = outputs["logits"][range(len(labels)),labels]
            outputs_c[range(len(labels)), labels] = torch.tensor(-float("inf"), device=outputs_c.device)
            largest_other = outputs_c.max(-1)[0]
            self.aum[indices] = assigned_logit - largest_other
            self.epoch_counting[indices] += 1
            torch.save(self.aum, os.path.join(self.args.output_dir, "aum_{}_{}.pt".format(self.flip_name,self.epoch_counting.min().item())))

            # filter the data
            # if self.current_epoch == self.end_epoch and self.filter:

            current_epoch=self.current_step*self.args.per_device_train_batch_size/len(self.train_dataset)
            self.current_step+=1

            if int(current_epoch*100) == self.end_epoch and self.filter:
                aum_scores=self.aum[self.flip_index].cpu().detach().numpy()
                threshold = np.quantile(aum_scores, 0.99)

                remove_index = []
                for i in tqdm(range(len(self.aum))):
                    if i in self.flip_index:
                        continue
                    elif self.aum[i] < threshold:
                        remove_index.append(i)
                save_dir = os.path.join(self.args.output_dir, "filtered_index_{}.txt".format(self.flip_name))
                with open(save_dir, "w") as fp:
                    json.dump(remove_index, fp)

                self.current_step=0 # reset for eval set
                self.filter_set="done"
