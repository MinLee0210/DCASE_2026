import os
import pprint
import random
import argparse
import copy
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from easydict import EasyDict

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import BaseOptions
from src.data.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from src.pipelines.evaluate import eval_epoch, setup_model

from src.utils.basic_utils import (
    AverageMeter,
    save_checkpoint,
    rename_latest_to_best,
)
from src.utils.log_utils import write_log, WandbLogger
from src.utils.model_utils import count_parameters, ModelEMA, get_run_name

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def set_seed(seed, use_cuda=True):
    """Sets the random seed.

    Args:
        seed (int): Seed.
        use_cuda (bool): Use cuda.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    criterion,
    train_loader,
    optimizer,
    opt,
    epoch_i,
    scaler=None,
    wandb_logger=None,
):
    """Trains for one epoch.

    Args:
        model: Model.
        criterion: Criterion.
        train_loader: Train loader.
        optimizer: Optimizer.
        opt: Options.
        epoch_i (int): Epoch index.
        scaler: GradScaler for AMP. Defaults to None.
    """
    logger.info(f"[Epoch {epoch_i + 1}]")
    model.train()
    criterion.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    use_amp = opt.get("use_amp", False)
    device_type = (
        "cuda"
        if "cuda" in str(opt.device)
        else ("mps" if "mps" in str(opt.device) else "cpu")
    )

    # Configure autocast keyword args
    autocast_kwargs = {"device_type": device_type, "enabled": use_amp}
    if device_type in ["cpu", "mps"]:
        autocast_kwargs["dtype"] = torch.bfloat16

    num_training_examples = len(train_loader)
    for batch_idx, batch in tqdm(
        enumerate(train_loader), desc="Training Iteration", total=num_training_examples
    ):
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)

        optimizer.zero_grad()

        with torch.amp.autocast(**autocast_kwargs):
            outputs = (
                model(**model_inputs, targets=targets)
                if opt.model_name == "cg_detr"
                else model(**model_inputs)
            )
            loss_dict = criterion(outputs, targets)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict.keys()
                if k in criterion.weight_dict
            )

        if scaler is not None:
            scaler.scale(losses).backward()
            if opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if opt.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v) * criterion.weight_dict[k]
                if k in criterion.weight_dict
                else float(v)
            )

        if wandb_logger and (batch_idx + 1) % opt.get("log_interval", 10) == 0:
            global_step = epoch_i * num_training_examples + batch_idx
            step_logs = {f"step_{k}": float(v) for k, v in loss_dict.items()}
            step_logs["lr"] = optimizer.param_groups[0]["lr"]
            wandb_logger.log_metrics(step_logs, step=global_step, prefix="train")

    write_log(opt, epoch_i, loss_meters)
    if wandb_logger:
        wandb_logger.log_metrics(
            {k: v.avg for k, v in loss_meters.items()}, step=epoch_i + 1, prefix="train"
        )


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    """Trains the model.

    Args:
        model: Model.
        criterion: Criterion.
        optimizer: Optimizer.
        lr_scheduler: LR scheduler.
        train_dataset: Train dataset.
        val_dataset: Val dataset.
        opt: Options.
    """
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    save_submission_filename = "latest_{}_val_preds.jsonl".format(opt.dset_name)

    wandb_run_name = get_run_name(
        opt.model_name,
        opt.lr,
        opt.bsz,
        experiment_type=opt.get("experiment_type", "lcs_detr"),
        random_suffix=True,
        random_suffix_len=6,
    )
    wandb_logger = WandbLogger(
        project_name=opt.get("wandb_project", f"DCASE2026_{opt.dset_name}"),
        run_name=wandb_run_name,
        config=vars(opt),
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
    )

    if opt.model_ema:
        logger.info("Using model EMA...")
        model_ema = ModelEMA(model, decay=opt.ema_decay)

    # Setup AMP GradScaler for CUDA if enabled
    scaler = (
        torch.amp.GradScaler()
        if opt.get("use_amp", False) and "cuda" in str(opt.device)
        else None
    )

    prev_best_score = 0
    for epoch_i in trange(opt.n_epoch, desc="Epoch"):
        train_epoch(
            model, criterion, train_loader, optimizer, opt, epoch_i, scaler=scaler
        )
        lr_scheduler.step()

        if opt.model_ema:
            model_ema.update(model)

        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            with torch.no_grad():
                if opt.model_ema:
                    metrics, eval_loss_meters, latest_file_paths = eval_epoch(
                        model_ema.module,
                        val_dataset,
                        opt,
                        save_submission_filename,
                        criterion,
                    )
                else:
                    metrics, eval_loss_meters, latest_file_paths = eval_epoch(
                        model, val_dataset, opt, save_submission_filename, criterion
                    )

            write_log(opt, epoch_i, eval_loss_meters, metrics=metrics, mode="val")
            logger.info("metrics {}".format(pprint.pformat(metrics["brief"], indent=4)))
            if wandb_logger:
                val_logs = {k: v.avg for k, v in eval_loss_meters.items()}
                if "brief" in metrics:
                    val_logs.update(metrics["brief"])
                wandb_logger.log_metrics(val_logs, step=epoch_i + 1, prefix="val")

            stop_score = metrics["brief"]["MR-full-R1@0.7"]

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                save_checkpoint(model, optimizer, lr_scheduler, epoch_i, opt)
                logger.info("The checkpoint file has been updated.")
                rename_latest_to_best(latest_file_paths)

    if wandb_logger:
        wandb_logger.finish()


def main(opt, resume=None):
    """Main function.

    Args:
        opt: Options.
        resume: Resume path.
    """
    logger.info("Setup config, data and model...")
    set_seed(opt.seed)

    # dataset & data loader
    dataset_config = EasyDict(
        data_path=opt.train_path,
        ctx_mode=opt.ctx_mode,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        a_feat_type=opt.a_feat_type,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=True,
    )

    train_dataset = StartEndDataset(**dataset_config)
    copied_eval_config = copy.deepcopy(dataset_config)
    copied_eval_config.data_path = opt.val_path
    eval_dataset = StartEndDataset(**copied_eval_config)

    # prepare model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)

    logger.info(f"Model {model}")
    count_parameters(model, verbose=True)

    if resume is not None:
        checkpoint = torch.load(resume, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info("Loaded model checkpoint: {}".format(resume))

    if opt.get("use_compile", False):
        logger.info("Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode="max-autotune")
            logger.info("Model compilation enabled successfully.")
        except Exception as e:
            logger.warning(
                f"Failed to compile model: {e}. Falling back to uncompiled model."
            )

    logger.info("Start Training...")

    # start training
    train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="config path")
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="specify model path for fine-tuning. If None, train the model from scratch.",
    )
    args = parser.parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    main(opt, args.resume)
