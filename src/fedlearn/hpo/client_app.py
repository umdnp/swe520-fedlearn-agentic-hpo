from __future__ import annotations

import logging

from flwr.app import Context
from flwr.clientapp import ClientApp
from flwr.common import Message, RecordDict, ArrayRecord, MetricRecord
from sklearn.pipeline import Pipeline

from fedlearn.common.config import HParams
from fedlearn.common.data_split import get_client_train_eval_by_key, CLIENT_KEYS
from fedlearn.common.metrics import compute_binary_metrics
from fedlearn.common.model import get_model, set_model_params, get_model_params

app = ClientApp()

logger = logging.getLogger(__name__)


def _get_client_key(context: Context) -> str:
    """
    Map Flower's partition-id to our logical client bucket name.
    """
    partition_id = int(context.node_config["partition-id"])
    try:
        return CLIENT_KEYS[partition_id]
    except IndexError:
        raise ValueError(
            f"partition-id={partition_id} out of range for CLIENT_KEYS={CLIENT_KEYS}"
        )


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
    Perform one round of local training for this client.

    Steps:
      1) Load this client's local train/eval data.
      2) Load global model parameters from the server.
      3) Initialize the local model with those parameters.
      4) Train for 'local-epochs' on the local training data.
      5) Compute metrics on the local training data.
      6) Return updated parameters and metrics to the server.
    """
    # load local train data (i.e. for this region)
    client_key = _get_client_key(context)
    X_train, y_train, _, _ = get_client_train_eval_by_key(client_key)

    hp = HParams.from_message(message, context)
    logger.info(f"[Client] Hyperparams this round: {hp}")

    model = _init_model(message, context, hp)
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]

    # local training
    X_proc = pre.transform(X_train)
    clf.fit(X_proc, y_train)  # uses max_iter=local_epochs

    # compute metrics on train split
    metrics_dict = compute_binary_metrics(model, X_train, y_train)
    num_examples = float(len(X_train))
    metrics_dict["num-examples"] = num_examples

    reply_content = RecordDict({
        "arrays": ArrayRecord(get_model_params(model)),
        "metrics": MetricRecord(metrics_dict),
    })

    reply_message = Message(
        content=reply_content,
        reply_to=message,
    )

    return reply_message


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    """
    Local evaluation using current global parameters.

    This uses the same partitioning logic and eval split as `train`,
    but does not perform any further training.
    """
    # load local eval data
    client_key = _get_client_key(context)
    _, _, X_eval, y_eval = get_client_train_eval_by_key(client_key)

    model = _init_model(message, context)

    # compute metrics on eval split
    metrics_dict = compute_binary_metrics(model, X_eval, y_eval)
    num_examples = float(len(X_eval))
    metrics_dict["num-examples"] = num_examples

    reply_content = RecordDict({
        "metrics": MetricRecord(metrics_dict),
    })

    return Message(
        content=reply_content,
        reply_to=message,
    )
