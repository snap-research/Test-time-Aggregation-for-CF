import torch
import copy

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
    ):

        self.patience = early_stopping_patience
        self.counter = 0
        self.best_score = -float("inf")
        self.best_state_dict = None

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
            logger.info(
                "Saving Model... New best score: {:.4}".format(
                    current_score
                )
            )
            self.best_state_dict = copy.deepcopy(model.state_dict())

        # we should do early stopping if the counter is greater than the patience and early stop is enabled
        if self.counter > self.patience:
            logger.info("Training Stopped. Early stopping activated. ")
            return True
        else:
            return False