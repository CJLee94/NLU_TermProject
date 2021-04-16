import torch

class Trainer:
    def __init__(self, cfg):
        self.tr_data, self.ts_data = get_dataset(cfg.DATASET)
        self.model = get_model(cfg.MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        self.epoch = cfg.TRAIN.EPOCH
        self.model.to(self.device)

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        param_optimizer = list(self.model.named_parameters())

        self.optimizer = get_optimizer(param_optimizer, cfg.OPTIMIZER)

        self.tr_loader = Dataloader(self.dataset, sampler=train_sampler, batch_size=cfg.DATASET.BATCHSIZE)

    def train(self):
        global_step = 0
        nb_tr_step = 0
        for e in range(int(self.epoch)):
            tr_loss = 0.0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(self.tr_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids)

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1

                if (step + 1) % 20 == 0:
                    print(tr_loss / nb_tr_steps / nb_tr_examples)
                    tr_loss = 0
                    nb_tr_examples, nb_tr_steps = 0, 0

            self.valid()

    def valid(self):
        for step, batch in enumerate(self.ts_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = self.model(input_ids, segment_ids, input_mask, label_ids)




