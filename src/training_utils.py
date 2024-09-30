import copy
import os

import torch

from src.constants import DEFAULT_MODEL_CKPT_PATH


class EarlyStopper:
    """Early stopper class to stop training when the model is not improving

    Parameters
    ----------
    patience: int
        The number of evaluations to wait before stopping the training
    greater_better: bool
        Whether the metric is better when it is greater
    early_stop: bool
        Whether to stop the training early
    metric: str
        The metric to evaluate the model
    """

    def __init__(
        self,
        early_stopping_patience: int,
        model_type: str,
        loss_function: str,
        dataset: str,
    ) -> None:

        self.patience = early_stopping_patience
        self.counter = 0
        self.best_score = -float("inf")
        self.best_state_dict = None
        self.model_name = f"{dataset}_{model_type}_{loss_function}.ckpt"

    def __call__(
        self,
        current_score: float,
        model: torch.nn.Module,
        logger: int,
    ) -> bool:

        if current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter == 0:
            logger.info("Saving Model... New best score: {:.4}".format(current_score))
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.dump_ckpt(model_type=self.model_name)

        # we should do early stopping if the counter is greater than the patience and early stop is enabled
        if self.counter > self.patience:
            logger.info("Training Stopped. Early stopping activated. ")
            return True
        else:
            return False

    def dump_ckpt(
        self, model_type: str, ckpt_path: str = DEFAULT_MODEL_CKPT_PATH
    ) -> None:

        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(self.best_state_dict, f"{ckpt_path}/{model_type}")
