import argparse

import pytorch_lightning as pl
import torch
import warnings

warnings.filterwarnings("ignore")

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from myutils.tools import color_print, to_list

from config import get_cfg_defaults
from data.make_dataset import make_data
from models import make_model
from utils import (
    clear_folder,
    build_logger,
    get_ckpt_path,
    make_callbacks,
    write_model_summary,
)

ROOT_DIR = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default="[32, 64, 64, 128]")
    parser.add_argument("--nblocks", type=str, default="[1,1,3,1]")
    parser.add_argument("--ablation", type=str, default=None)


    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--grad", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--min_epoch", type=int, default=1)
    parser.add_argument("--use_profiler", type=int, default=0)
    parser.add_argument("--use_lr_find", type=int, default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("-t", "--test", type=int, default=0)
    parser.add_argument("-l", "--log", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--theme", type=str, default="best")
    parser.add_argument("--collect", type=int, default=0)
    parser.add_argument("--clear_log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_as_val", type=int, default=999)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--test_noise_level", type=int, default=30)
    parser.add_argument("--test_noise_type", type=str, default="bg")
    args = parser.parse_args()

    if args.seed != 42:
        pl.seed_everything(args.seed)

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    if args.batch_size > 0:
        cfg.DATASET.batch_size = args.batch_size
    ds, dl = make_data(cfg.DATASET, args=args)

    args.profiler = (
        pl.profilers.SimpleProfiler(dirpath="./", filename="test")
        if args.use_profiler
        else None
    )

    # print(str(dict(cfg)))
    model = make_model(args.cfg, cfg, args)
    callbacks = make_callbacks(args, cfg)

    trainer = pl.Trainer(
        max_epochs=2 if args.use_profiler else cfg.MODEL.epochs,
        accelerator="gpu",
        devices=args.gpu,
        logger=build_logger(args, ROOT_DIR),
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        default_root_dir=ROOT_DIR,
        strategy="ddp_find_unused_parameters_true" if len(args.gpu) > 1 else "auto",
        profiler=args.profiler,
        enable_checkpointing=False if args.use_profiler else True,
        limit_train_batches=100 if args.use_profiler else 1.0,
        limit_val_batches=100 if args.use_profiler else 1.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.grad,
        precision="16-mixed" if args.precision == 16 else "32",
    )

    color_print(f"logger path : {trainer.logger.log_dir}")
    log_dir = trainer.logger.log_dir

    if args.test == 0 and args.clear_log:
        clear_folder(log_dir)

    if not args.test:
        ckpt_path = get_ckpt_path(log_dir, theme="last") if args.resume else None

        # val_dl = to_list(dl.test)[1]
        val_dl = dl.val
        if args.test_as_val != 999:
            val_dl = to_list(dl.test)[args.test_as_val]

        trainer.fit(model, dl.train, val_dataloaders=val_dl, ckpt_path=ckpt_path)

        write_model_summary(model, log_dir)

    else:
        ckpt_path = get_ckpt_path(log_dir, theme=args.theme)
        trainer.trainset_wo_transform = dl.train_wo_transform

        if not args.collect:
            for test_dl in to_list(dl.test):
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
        else:
            for test_dl in to_list(dl.test) + [dl.val]:
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
