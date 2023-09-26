import torch
import logging
import speechbrain as sb
import torch.nn.functional as F
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import LoopedLoader

# Logger info
logger = logging.getLogger(__name__)

class KWSBrain(sb.Brain):
    agg_metric = "average"

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig # B, L

        return self.modules.model(wavs)

    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        uttid = batch.id
        targets, lens = batch.target # B, 1
        targets = targets.squeeze()

        # Compare the waveforms
        loss = self.hparams.loss(predictions, targets)
        if (stage != sb.Stage.TRAIN):
            self.error_metric.append(
                uttid,
                probabilities=predictions.unsqueeze(0),
                targets=targets
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=self.hparams.loss
        )

        # Add a metric for evaluation sets
        if stage != sb.Stage.TRAIN:
            self.error_metric = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        if stage != sb.Stage.TRAIN:
            # Summarize the statistics from the stage for record-keeping.
            stats = {
                "loss": stage_loss,
                "error_metric": self.error_metric.summarize(self.agg_metric),
            }

        # At the end of validation, we can write stats, checkpoints and update LR.
        if (stage == sb.Stage.VALID):
            current_lr, next_lr = self.hparams.lr_scheduler(epoch)
            schedulers.update_learning_rate(self.optimizer, next_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch": epoch, "LR": current_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best task1_metric
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error_metric"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Test Stage": 'Loss and Error statistics of the model'},
                test_stats=stats,
            )

    def evaluate_robustness(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={}
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )

        # Recover best checkpoint for evaluation
        if self.hparams.attacker_checkpointer is not None:
            self.hparams.attacker_checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )

        if self.hparams.victim_checkpointer is not None:
            self.hparams.victim_checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )

        # Attack the model
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0

        for batch in tqdm(
            test_set,
            dynamic_ncols=True,
            disable=not progressbar,
            colour="CYAN",
        ):
            self.step += 1
            batch = batch.to(self.device)
            # Create the adversarial samples
            wavs, lens = batch.sig
            targets, lens = batch.target
            targets = targets.squeeze()
            adversarial_samples, _ = self.modules.attack(wavs, targets)

            # Batch everything together
            batch.sig = sb.dataio.batch.PaddedData(
                data=adversarial_samples,
                lengths=batch.sig[1]
            )

            with torch.no_grad():
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

        # Only run evaluation "on_stage_end" on main process
        run_on_main(
            self.on_stage_end, args=[sb.Stage.TEST, avg_test_loss, None]
        )

        self.step = 0
        return avg_test_loss
