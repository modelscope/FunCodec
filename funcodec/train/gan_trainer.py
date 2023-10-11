# Copyright 2021 Tomoki Hayashi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Zhihao Du for GAN-based Codec models.

"""Trainer module for GAN-based training."""

import argparse
import dataclasses
import logging
import time

from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
import os
import soundfile
from io import BytesIO

from typeguard import check_argument_types

from funcodec.schedulers.abs_scheduler import AbsBatchStepScheduler
from funcodec.schedulers.abs_scheduler import AbsScheduler
from funcodec.torch_utils.device_funcs import to_device
from funcodec.torch_utils.recursive_op import recursive_average
from funcodec.train.distributed_utils import DistributedOption
from funcodec.train.reporter import SubReporter
from funcodec.train.trainer import Trainer
from funcodec.train.trainer import TrainerOptions
from funcodec.utils.build_dataclass import build_dataclass
from funcodec.utils.types import str2bool
import gc

if torch.distributed.is_available():
    from torch.distributed import ReduceOp

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield

    GradScaler = None

try:
    import fairscale
except ImportError:
    fairscale = None


@dataclasses.dataclass
class GANTrainerOptions(TrainerOptions):
    """Trainer option dataclass for GANTrainer."""

    generator_first: bool
    disc_grad_clip: float
    disc_grad_clip_type: float
    gen_train_interval: int
    disc_train_interval: int
    sampling_rate: int


class GANTrainer(Trainer):
    """Trainer for GAN-based training.

    If you'd like to use this trainer, the model must inherit
    espnet.train.abs_gan_espnet_model.AbsGANESPnetModel.

    """

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()."""
        assert check_argument_types()
        return build_dataclass(GANTrainerOptions, args)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add additional arguments for GAN-trainer."""
        parser.add_argument(
            "--generator_first",
            type=str2bool,
            default=False,
            help="Whether to update generator first.",
        )

    @classmethod
    def train_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
        **kwargs,
    ) -> Tuple[bool, bool]:
        """Train one epoch."""
        assert check_argument_types()

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        disc_grad_clip = options.disc_grad_clip
        grad_clip_type = options.grad_clip_type
        disc_grad_clip_type = options.disc_grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        generator_first = options.generator_first
        distributed = distributed_option.distributed
        gen_train_interval = options.gen_train_interval
        disc_train_interval = options.disc_train_interval

        # Check unavailable options
        # TODO(kan-bayashi): Support the use of these options
        if accum_grad > 1:
            raise NotImplementedError(
                "accum_grad > 1 is not supported in GAN-based training."
            )
        if grad_noise:
            raise NotImplementedError(
                "grad_noise is not supported in GAN-based training."
            )

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        all_steps_are_invalid = True
        max_update_stop = False
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (_, batch_org) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch_org, dict), type(batch_org)

            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch_org, "cuda" if ngpu > 0 else "cpu")
            del batch_org
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            turn_start_time = time.perf_counter()
            turns = []
            if generator_first:
                if iiter % gen_train_interval == 0:
                    turns.append("generator")
                if iiter % disc_train_interval == 0:
                    turns.append("discriminator")
            else:
                if iiter % disc_train_interval == 0:
                    turns.append("discriminator")
                if iiter % gen_train_interval == 0:
                    turns.append("generator")
            for turn in turns:
                with autocast(scaler is not None):
                    with reporter.measure_time(f"{turn}_forward_time"):
                        retval = model(turn == "generator", batch)

                        # Note(kamo):
                        # Supporting two patterns for the returned value from the model
                        #   a. dict type
                        if isinstance(retval, dict):
                            loss = retval["loss"]
                            stats = retval["stats"]
                            weight = retval["weight"]
                            optim_idx = retval.get("optim_idx")
                            if optim_idx is not None and not isinstance(optim_idx, int):
                                if not isinstance(optim_idx, torch.Tensor):
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {type(optim_idx)}"
                                    )
                                if optim_idx.dim() >= 2:
                                    raise RuntimeError(
                                        "optim_idx must be int or 1dim torch.Tensor, "
                                        f"but got {optim_idx.dim()}dim tensor"
                                    )
                                if optim_idx.dim() == 1:
                                    for v in optim_idx:
                                        if v != optim_idx[0]:
                                            raise RuntimeError(
                                                "optim_idx must be 1dim tensor "
                                                "having same values for all entries"
                                            )
                                    optim_idx = optim_idx[0].item()
                                else:
                                    optim_idx = optim_idx.item()

                        # b. tuple or list type
                        else:
                            raise RuntimeError("model output must be dict.")

                    stats = {k: v for k, v in stats.items() if v is not None}
                    if ngpu > 1 or distributed:
                        # Apply weighted averaging for loss and stats
                        loss = (loss * weight.type(loss.dtype)).sum()

                        # if distributed, this method can also apply all_reduce()
                        stats, weight = recursive_average(stats, weight, distributed)

                        # Now weight is summation over all workers
                        loss /= weight

                    if distributed:
                        # NOTE(kamo): Multiply world_size since DistributedDataParallel
                        # automatically normalizes the gradient by world_size.
                        loss *= torch.distributed.get_world_size()

                reporter.register(stats, weight)

                with reporter.measure_time(f"{turn}_backward_time"):
                    if scaler is not None:
                        # Scales loss.  Calls backward() on scaled loss
                        # to create scaled gradients.
                        # Backward passes under autocast are not recommended.
                        # Backward ops run in the same dtype autocast chose
                        # for corresponding forward ops.
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    for iopt, optimizer in enumerate(optimizers):
                        if optim_idx is not None and iopt != optim_idx:
                            continue
                        scaler.unscale_(optimizer)

                # TODO(kan-bayashi): Compute grad norm without clipping
                grad_norm = None
                if turn == "generator":
                    _grad_clip, _grad_clip_type = grad_clip, grad_clip_type
                else:
                    _grad_clip, _grad_clip_type = disc_grad_clip, disc_grad_clip_type
                if _grad_clip > 0.0:
                    # compute the gradient norm to check if it is normal or not
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=_grad_clip,
                        norm_type=_grad_clip_type,
                    )
                    # PyTorch<=1.4, clip_grad_norm_ returns float value
                    if not isinstance(grad_norm, torch.Tensor):
                        grad_norm = torch.tensor(grad_norm)

                if grad_norm is None or torch.isfinite(grad_norm):
                    all_steps_are_invalid = False
                    with reporter.measure_time(f"{turn}_optim_step_time"):
                        for iopt, (optimizer, scheduler) in enumerate(
                            zip(optimizers, schedulers)
                        ):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            if scaler is not None:
                                # scaler.step() first unscales the gradients of
                                # the optimizer's assigned params.
                                scaler.step(optimizer)
                                # Updates the scale for next iteration.
                                scaler.update()
                            else:
                                optimizer.step()
                            if isinstance(scheduler, AbsBatchStepScheduler):
                                scheduler.step()
                else:
                    logging.warning(
                        f"The grad norm is {grad_norm}. " "Skipping updating the model."
                    )
                    # Must invoke scaler.update() if unscale_() is used in the
                    # iteration to avoid the following error:
                    #   RuntimeError: unscale_() has already been called
                    #   on this optimizer since the last update().
                    # Note that if the gradient has inf/nan values,
                    # scaler.step skips optimizer.step().
                    if scaler is not None:
                        for iopt, optimizer in enumerate(optimizers):
                            if optim_idx is not None and iopt != optim_idx:
                                continue
                            scaler.step(optimizer)
                            scaler.update()

                for iopt, optimizer in enumerate(optimizers):
                    # NOTE(kan-bayashi): In the case of GAN, we need to clear
                    #   the gradient of both optimizers after every update.
                    optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    {
                        f"optim{optim_idx}_lr{i}": pg["lr"]
                        for i, pg in enumerate(optimizers[optim_idx].param_groups)
                        if "lr" in pg
                    },
                )
                reporter.register(
                    {f"{turn}_train_time": time.perf_counter() - turn_start_time}
                )
                turn_start_time = time.perf_counter()

            reporter.register({"train_time": time.perf_counter() - start_time})
            start_time = time.perf_counter()

            if distributed:
                if hasattr(model.module, "num_updates"):
                    model.module.set_num_updates(model.module.get_num_updates() + 1)
                    options.num_updates = model.module.get_num_updates()
                    if model.module.get_num_updates() >= options.max_update:
                        max_update_stop = True
            else:
                if hasattr(model, "num_updates"):
                    model.set_num_updates(model.get_num_updates() + 1)
                    options.num_updates = model.get_num_updates()
                    if model.get_num_updates() >= options.max_update:
                        max_update_stop = True

            to_save_model = model.module if distributed else model
            if (hasattr(options, "num_updates") and
                options.save_ckpt_every_steps > 0 and
                options.num_updates % options.save_ckpt_every_steps == 0):
                if options.use_pai:
                    buffer = BytesIO()
                    torch.save(
                        {
                            "model": to_save_model.state_dict(),
                            "optimizers": [o.state_dict() for o in optimizers],
                            "reporter": kwargs["par_reporter"].state_dict(),
                            "schedulers": [
                                s.state_dict() if s is not None else None
                                for s in schedulers
                            ],
                            "scaler": scaler.state_dict() if scaler is not None else None,
                        },
                        buffer,
                    )
                    options.oss_bucket.put_object(
                        os.path.join(options.output_dir, f"{options.num_updates}steps.ckpt"),
                        buffer.getvalue())
                else:
                    torch.save(
                        {
                            "model": to_save_model.state_dict(),
                            "optimizers": [o.state_dict() for o in optimizers],
                            "reporter": kwargs["par_reporter"].state_dict(),
                            "schedulers": [
                                s.state_dict() if s is not None else None
                                for s in schedulers
                            ],
                            "scaler": scaler.state_dict() if scaler is not None else None,
                        },
                        os.path.join(options.output_dir, f"{options.num_updates}steps.ckpt"),
                    )
                ckpt_path = os.path.join(options.output_dir, f"{options.num_updates}steps.ckpt")
                logging.info(f"Saved checkpoint to {ckpt_path}.")

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                num_updates = options.num_updates if hasattr(options, "num_updates") else None
                logging.info(reporter.log_message(-log_interval, num_updates=num_updates))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

            del batch, retval
            gc.collect()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid, max_update_stop

    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: GANTrainerOptions,
        distributed_option: DistributedOption,
    ) -> None:
        """Validate one epoch."""
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = distributed_option.distributed
        generator_first = options.generator_first

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        cached_retval = None
        for (_, batch_org) in iterator:
            assert isinstance(batch_org, dict), type(batch_org)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch_org, "cuda" if ngpu > 0 else "cpu")
            del batch_org
            if no_forward_run:
                continue

            if generator_first:
                turns = ["generator", "discriminator"]
            else:
                turns = ["discriminator", "generator"]
            for turn in turns:
                retval = model(turn == "generator", batch)
                if turn == "generator":
                    cached_retval = retval
                if isinstance(retval, dict):
                    stats = retval["stats"]
                    weight = retval["weight"]
                else:
                    _, stats, weight = retval
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for stats.
                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)
                reporter.register(stats, weight)

            reporter.next()

            del batch, retval
            gc.collect()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        def write_wav(file_path, wav, sr=16_000):
            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                wav = wav / np.max(np.abs(wav)) * (2 ** 15) * 0.9
                if options.use_pai:
                    buffer = BytesIO()
                    soundfile.write(buffer, wav.astype(np.int16),
                                    sr, "PCM_16", "LITTLE", "WAV", True)
                    options.oss_bucket.put_object(file_path, buffer.getvalue())
                else:
                    if not os.path.exists(os.path.dirname(file_path)):
                        os.makedirs(os.path.dirname(file_path))
                    soundfile.write(file_path, wav.astype(np.int16),
                                    sr, "PCM_16", "LITTLE", "WAV", True)

        # process for extra retval
        if cached_retval is not None:
            if isinstance(cached_retval, dict):
                iepoch = reporter.get_epoch()
                for name in ["real", "fake"]:
                    if name in cached_retval:
                        speech = cached_retval[name].squeeze(1).detach().cpu().numpy()
                        for i in range(min(speech.shape[0], 5)):
                            wav_path = os.path.join(options.output_dir, "wavs",
                                                    "epoch{:04d}".format(iepoch),
                                                    "ex{}_{}.wav".format(i+1, name))
                            write_wav(wav_path, speech[i], sr=options.sampling_rate)

                        del speech
