import time
import json
import wandb


def write_log(opt, epoch_i: int, loss_meters, metrics=None, mode="train"):
    # log
    if mode == "train":
        to_write = opt.train_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i + 1,
            loss_str=" ".join(
                ["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]
            ),
        )
        filename = opt.train_log_filepath
    else:
        to_write = opt.eval_log_txt_formatter.format(
            time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
            epoch=epoch_i,
            loss_str=" ".join(
                ["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]
            ),
            eval_metrics_str=json.dumps(metrics),
        )
        filename = opt.eval_log_filepath

    with open(filename, "a") as f:
        f.write(to_write)


class WandbLogger:
    def __init__(
        self, project_name: str, run_name: str, config: dict = None, entity: str = None
    ):
        """
        Initializes the W&B run.
        """
        self.run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True,
        )
        self.best_accuracy = 0.0

    def log_metrics(self, metrics, step, prefix="eval"):
        """
        Logs a dictionary of metrics.

        Args:
            metrics (dict): The dictionary from your eval function.
            step (int): Current global step or epoch.
            prefix (str): Dashboard grouping (e.g., 'train' or 'eval').
        """
        # Format keys: {'loss': 0.5} -> {'eval/loss': 0.5}
        log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Log to W&B
        self.run.log(log_dict, step=step)

        # Optional: Track Best Metric Logic
        if "accuracy" in metrics:
            if metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = metrics["accuracy"]
                self.run.summary["best_accuracy"] = self.best_accuracy
                print(f"New best accuracy: {self.best_accuracy:.4f}")

    def log_artifact(self, model_path, name="model-checkpoint"):
        """Save your model file to W&B"""
        artifact = wandb.Artifact(name, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def finish(self):
        """Closes the W&B run"""
        self.run.finish()
