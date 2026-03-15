from __future__ import annotations

import logging

import pandas as pd
from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.common import ArrayRecord, Message, MetricRecord, RecordDict
from sklearn.pipeline import Pipeline

from fedlearn.common.config import DataSplit, HParams, CONFIG_KEY, TRAIN_SPLIT, EVAL_SPLIT
from fedlearn.common.data_split import CLIENT_KEYS, get_client_train_val_test_by_key
from fedlearn.common.metrics import compute_binary_metrics
from fedlearn.common.model import get_model, get_model_params, set_model_params

app = ClientApp()

logger = logging.getLogger(__name__)


def _get_client_key(context: Context) -> str:
    """
    Map Flower's partition-id to our logical client bucket name.
    """
    partition_id = int(context.node_config["partition-id"])

    try:
        return CLIENT_KEYS[partition_id]
    except IndexError as ex:
        raise ValueError(
            f"partition-id={partition_id} out of range for CLIENT_KEYS={CLIENT_KEYS}"
        ) from ex


def _get_cfg_value(message: Message, context: Context, key: str, default: str) -> str:
    """
    Read a config value from the incoming message or fallback to run_config.
    """
    cfg = message.content.get(CONFIG_KEY)

    if cfg is not None and key in cfg:
        return str(cfg[key])

    return str(context.run_config.get(key, default))


def _get_train_split(message: Message, context: Context) -> DataSplit:
    """
    Determine which dataset split should be used for training.
    """
    value = _get_cfg_value(message, context, TRAIN_SPLIT, DataSplit.TRAIN.value)

    try:
        return DataSplit(value)
    except ValueError as ex:
        raise ValueError(f"Unknown train split: {value!r}") from ex


def _get_eval_split(message: Message, context: Context) -> DataSplit:
    """
    Determine which dataset split should be used for evaluation.
    """
    value = _get_cfg_value(message, context, EVAL_SPLIT, DataSplit.TEST.value)

    try:
        return DataSplit(value)
    except ValueError as ex:
        raise ValueError(f"Unknown eval split: {value!r}") from ex


def _init_model(message: Message, context: Context, hp: HParams | None = None) -> Pipeline:
    """
    Build model and load incoming model params.
    """
    incoming_arrays = message.content["arrays"]

    if hp is None:
        hp = HParams.from_message(message, context)

    model = get_model(hp)
    set_model_params(model, incoming_arrays.to_numpy_ndarrays())

    return model


@app.train()
def train(message: Message, context: Context) -> Message:
    """
    Perform one round of local training.

    TRAIN_SPLIT determines which dataset is used:
    - TRAIN: fit on local train split
    - TRAIN_VAL: fit on local train + validation splits
    """
    client_key = _get_client_key(context)
    X_train, y_train, X_val, y_val, _, _ = get_client_train_val_test_by_key(client_key)

    train_split = _get_train_split(message, context)

    if train_split == DataSplit.TRAIN:
        X_fit, y_fit = X_train, y_train
    elif train_split == DataSplit.TRAIN_VAL:
        X_fit = pd.concat([X_train, X_val], axis=0, ignore_index=True)
        y_fit = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    else:
        raise ValueError(f"Unsupported training split for train(): {train_split!r}")

    hp = HParams.from_message(message, context)
    logger.info("[Client] Hyperparams this round: %s, train_split=%s", hp, train_split.value)

    model = _init_model(message, context, hp)
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # local training
    X_proc = pre.transform(X_fit)
    clf.fit(X_proc, y_fit)  # uses max_iter=local_epochs

    # compute metrics on the local fit dataset
    metrics_dict = compute_binary_metrics(model, X_fit, y_fit)
    metrics_dict["num-examples"] = float(len(X_fit))

    reply_content = RecordDict({
        "arrays": ArrayRecord(get_model_params(model)),
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(content=reply_content, reply_to=message)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """
    Perform local evaluation.

    EVAL_SPLIT determines which dataset is used:
    - VALIDATION: evaluate on local validation split
    - TEST: evaluate on local test split
    """
    client_key = _get_client_key(context)
    _, _, X_val, y_val, X_test, y_test = get_client_train_val_test_by_key(client_key)

    eval_split = _get_eval_split(message, context)

    if eval_split == DataSplit.VALIDATION:
        X_eval, y_eval = X_val, y_val
    elif eval_split == DataSplit.TEST:
        X_eval, y_eval = X_test, y_test
    else:
        raise ValueError(f"Unsupported evaluation split for evaluate(): {eval_split!r}")

    model = _init_model(message, context)

    # compute metrics on the evaluation split
    metrics_dict = compute_binary_metrics(model, X_eval, y_eval)
    metrics_dict["num-examples"] = float(len(X_eval))

    reply_content = RecordDict({
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(content=reply_content, reply_to=message)
