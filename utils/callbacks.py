from myutils.torch.lightning.callbacks import (
    Color_progress_bar,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from myutils.torch.lightning.callbacks.metrics import (
    BinaryACC_Callback,
    BinaryAUC_Callback,
)
from myutils.torch.lightning.callbacks import Collect_Callback


def common_callbacks():
    callbacks = [
        Color_progress_bar(),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
    ]

    return callbacks


def custom_callbacks(args, cfg):
    callbacks = []
    return callbacks


def training_callbacks(args):

    monitor = "val-auc"
    es = EarlyStopping

    callbacks = [
        # save last ckpt
        ModelCheckpoint(
            dirpath=None, save_top_k=0, save_last=True, save_weights_only=False
        ),
        # save best ckpt
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor=monitor,
            mode="max",
            save_last=False,
            filename="best-{epoch}-{val-auc:.4f}",
            save_weights_only=True,
            verbose=True,
        ),
    ]

    if args.earlystop:
        callbacks.append(
            es(
                monitor=monitor,
                min_delta=0.001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="max",
                stopping_threshold=0.998 if monitor == "val-auc+++val-acc" else 0.999,
                verbose=True,
            )
        )
    return callbacks


def make_collect_callbacks(args, cfg):
    name = args.cfg.replace("/", "-")
    callbacks = [
        Collect_Callback(
            batch_keys=["label", "vocoder_label"],
            output_keys=["feature"],
            save_path=f"./0-实验结果/npz/{name}",
        )
    ]
    return callbacks


def make_callbacks(args, cfg):
    callbacks = common_callbacks()
    callbacks += custom_callbacks(args, cfg)
    if not args.test:
        callbacks += training_callbacks(args)

    if args.collect and args.test:
        callbacks += make_collect_callbacks(args, cfg)

    return callbacks
