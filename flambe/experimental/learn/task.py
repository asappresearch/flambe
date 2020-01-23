class Task(Component):


    @staticmethod
    def dataloaders(config):

    @staticmethod:
    def models(config):

    @staticmethod
    def optimizers(models, config):

    @staticmethod
    def val_metric(metrics):


    def step() -> :
        self.trainer.step()
        self.trainer.train()
        self.trainer.validate()

    def metric():
        self.train.

    @staticmethod
    def train_batch(batch, model, criterion, device):
        """Returns the loss"""
        source, target = batch
        model.


    @staticmethod
    def train_step(model, dataloader, criterion, optimizers, config):
        model.train()

        data_iterator = iter(data_loader)
        # Zero the gradients and clear the accumulated loss
        optimizer.zero_grad()
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation):
            # Get next batch
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)

            # Compute loss
            loss = criterion(batch) / config.gradient_accumulation
            accumulated_loss += loss.item()
            loss.backward()

        # Log loss
        global_step = (self.iter_per_step * self._step) + i

        # Clip gradients if necessary
        if config.max_grad_norm:
            clip_grad_norm_(model.parameters(), max_grad_norm)
        if config.max_grad_abs_val:
            clip_grad_value_(model.parameters(), max_grad_abs_val)

        # Optimize
        optimizer.step()

        # Update iter scheduler
        if iter_scheduler is not None:
            learning_rate = self.iter_scheduler.get_lr()[0]  # type: ignore
            log(f'{tb_prefix}Training/LR', learning_rate, global_step)
            iter_scheduler.step()  # type: ignore

        # Zero the gradients when exiting a train step
        optimizer.zero_grad()

        log('Training/Loss', accumulated_loss, global_step)
        log('Training/Gradient_Norm', model.gradient_norm, global_step)
        log('Training/Parameter_Norm', model.parameter_norm, global_step)

    @staticmethod
    def val_step(models, dataloader, criterion, config):
        model.eval()

        with torch.no_grad():

            preds, targets = self._aggregate_preds(dataloader)
            val_loss = self.loss_fn(preds, targets).item()
            val_metric = self.metric_fn(preds, targets).item()

        # Update best model
        sign = (-1)**(self.lower_is_better)
        if self._best_metric is None or (sign * val_metric > sign * self._best_metric):
            self._best_metric = val_metric
            best_model_state = self.model.state_dict()
            for k, t in best_model_state.items():
                best_model_state[k] = t.cpu().detach()
            self._best_model = best_model_state

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step(val_loss)

        # Log metrics
        log(f'{tb_prefix}Validation/Loss', val_loss, self._step)
        log(f'{tb_prefix}Validation/{self.metric_fn}', val_metric, self._step)
        log(f'{tb_prefix}Best/{self.metric_fn}', self._best_metric, self._step)  # type: ignore
        for metric in self.extra_validation_metrics:
            log(f'{tb_prefix}Validation/{metric}',
                metric(preds, targets).item(), self._step)  # type: ignore

        return dict()