import sys
from typing import Dict

import dgl
import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import TAGCF_SEARCH_SPACE
from src.data import DatasetClass, get_dataloader
from src.eval_metrics import MetricClass
from src.loss_function import LossFunction
from src.model import ModelClass
from src.training_utils import EarlyStopper
from src.utils import init_logger, load_yaml, pre_process_graph


class MFPipeline:
    """
    matrix factorization pipeline

    Parameters
    ----------
    general_yaml_path: str
        the path to general yaml config file
    model_yaml_path: str
        the path to model-specific yaml config file
    """

    def __init__(
        self,
        general_yaml_path: str,
        model_yaml_path: str,
    ):

        self.load_config(
            general_yaml_path=general_yaml_path, model_yaml_path=model_yaml_path
        )

        self.load_dataset(self.config["dataset"])

        self.model = (
            ModelClass[self.config["model_type"]]
            .value(
                number_of_users=self.dataset.n_user,
                number_of_items=self.dataset.n_item,
                embedding_dim=self.config["embedding_dim"],
                loss_function=LossFunction[self.config["loss_function"]].value(),
                num_layers=self.config["n_layers"],
                embedding_table_weight=self.config.get("embedding_table_weight", None),
                m=self.config.get("m", None),
                n=self.config.get("n", None),
            )
            .to(self.device)
        )

        self.optimizer = torch.optim.Adam(
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            params=self.model.parameters(),
        )

        self.early_stopper = EarlyStopper(
            early_stopping_patience=self.config["early_stopping_patience"],
            model_type=self.config["model_type"],
            loss_function=self.config["loss_function"],
            dataset=self.config["dataset"],
        )

        self.eval_metrics = [
            MetricClass[metric].value(top_k=k)
            for metric in self.config["metrics"]
            for k in self.config["top_k"]
        ]
        self.logger = init_logger()

    def load_config(
        self,
        general_yaml_path: str,
        model_yaml_path: str,
    ) -> None:
        """
        load general configs and model-specific configs given their yaml files

        Parameters
        ----------
        general_yaml_path: str
            the path to general yaml config file
        model_yaml_path: str
            the path to model-specific yaml config file
        """

        self.config = load_yaml(path=general_yaml_path)
        self.device = self.config.get("device_id", "cpu")
        self.config["n_layers"] = self.config.get("n_layers", 0)
        self.config.update(load_yaml(path=model_yaml_path))

    def load_dataset(self, dataset_name: str) -> None:
        """
        get the dataset object given the data set name

        Parameters
        ----------
        dataset_name: str
            the string name of the dataset
        """
        self.dataset = DatasetClass[dataset_name].value()
        self.train_dataset = self.dataset.get_train()
        self.valid_dataset = self.dataset.get_valid()
        self.test_dataset = self.dataset.get_test()

    def get_dataloader(self, dataset: dgl.DGLGraph) -> DataLoader:
        """
        get the dataloader given the graph

        Parameters
        ----------
        dataset: dgl.DGLGraph
            the dgl graph object that the dataloader gets data from
        """

        return get_dataloader(
            graph=dataset,
            batch_size=self.config["batch_size"],
            num_users=self.dataset.n_user,
            num_items=self.dataset.n_item,
            num_workers=self.config["num_workers"],
            shuffle=True,
        )

    def handle_iteration(
        self,
        loss: torch.Tensor,
    ) -> None:
        """
        conduct backpropagatiopn given the loss

        Parameters
        ----------
        loss: torch.Tensor
            the loss of the model
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def eval(
        self,
        train_graph: dgl.DGLGraph,
        use_mp: bool = False,
        for_testing: bool = False,
        eval_batch_size: int = 1024,
    ) -> Dict[str, float]:
        """
        conducting evalution over either testing or validation set

        Parameters
        ----------
        train_graph: dgl.DGLGraph
            the dgl graph object to conduct message passing on
        use_mp: bool
            only works for TAGCF. it determins if we should use message passing for test-time aggregation
        for_testing: bool
            if we conduct evaluation on the testing set. (False for validation set)
        """
        # acquiring all embeddings for users and items
        self.model.eval()
        all_embeddings = self.model.get_all_embbedings(
            graph=train_graph, use_mp=use_mp
        ).squeeze()

        user_embeddings = all_embeddings[: self.dataset.n_user]
        item_embeddings = all_embeddings[self.dataset.n_user :]

        # masking positive interactions observed during the training
        eval_batch_num = (user_embeddings.shape[0] // eval_batch_size) + 1
        results = {
            f"{metric.__class__.__name__}@{metric.top_k}": 0
            for metric in self.eval_metrics
        }
        users, items = self.train_dataset.edges()
        items = items - self.dataset.n_user
        if for_testing:
            target_users, target_items = self.test_dataset.edges()
        else:
            target_users, target_items = self.valid_dataset.edges()
        target_items = target_items - self.dataset.n_user

        # conduct evaluation by splitting users into multiple batches
        # so that we don't blow up RAM, ideally this should be done using ANN
        for batch_idx in range(eval_batch_num):
            logits = torch.mm(
                user_embeddings[
                    batch_idx * eval_batch_size : (batch_idx + 1) * eval_batch_size
                ],
                item_embeddings.t(),
            )

            sample_mask = torch.mul(
                (users >= batch_idx * eval_batch_size),
                (users < (batch_idx + 1) * eval_batch_size),
            )
            users_in_this_batch, items_in_this_batch = (
                users[sample_mask],
                items[sample_mask],
            )

            # masking positive interactions observed during the training
            logits[
                users_in_this_batch - (batch_idx * eval_batch_size), items_in_this_batch
            ] = -torch.inf

            # accordingly load evaluation labels
            labels = torch.zeros_like(logits).bool()
            sample_mask = torch.mul(
                (target_users >= batch_idx * eval_batch_size),
                (target_users < (batch_idx + 1) * eval_batch_size),
            )
            users_in_this_batch, items_in_this_batch = (
                target_users[sample_mask],
                target_items[sample_mask],
            )
            labels[
                users_in_this_batch - (batch_idx * eval_batch_size), items_in_this_batch
            ] = True

            # masking out users without any interactions
            empty_row_mask = labels.sum(dim=1) != 0
            labels = labels[empty_row_mask]
            logits = logits[empty_row_mask]

            # preparing vars for metric calculation
            labels = labels.reshape(-1)
            indexes = torch.arange(0, logits.shape[0], device=self.device)
            expanded_indexes = (
                indexes.unsqueeze(-1)
                .expand(logits.shape[0], logits.shape[1])
                .reshape(-1)
            )

            for metric in self.eval_metrics:
                metric_name = f"{metric.__class__.__name__}@{metric.top_k}"

                results[metric_name] += (
                    metric(
                        preds=logits.reshape(-1),
                        target=labels,
                        indexes=expanded_indexes,
                    ).item()
                    * logits.shape[0]
                )
                metric.reset()

        for metric in self.eval_metrics:
            metric_name = f"{metric.__class__.__name__}@{metric.top_k}"
            results[metric_name] /= self.dataset.n_user
        self.model.train()
        return results

    def train(self) -> None:
        """
        the training logic for all matrix factorization models
        """
        train_dataloader = self.get_dataloader(
            dataset=self.train_dataset,
        )
        train_dataset = pre_process_graph(self.train_dataset)
        train_dataset = train_dataset.to(self.device)

        for epoch in range(self.config["total_epochs"]):
            self.logger.info(f"Training on Epoch: {epoch}")
            for batch in train_dataloader:
                batch = batch.to(self.device)
                model_output, loss = self.model(
                    graph=train_dataset,
                    user_ids=batch[:, 0],
                    positive_item_ids=batch[:, 1],
                    negative_item_ids=batch[:, 2],
                    is_training=True,
                )

                self.handle_iteration(loss=loss)

            # conduct eval every eval_steps epochs
            if (epoch + 1) % self.config["eval_steps"] == 0:
                eval_metrics = self.eval(train_graph=train_dataset, for_testing=False)
                self.logger.info(eval_metrics)
                if self.early_stopper(
                    current_score=eval_metrics[self.config["early_stopping_metric"]],
                    model=self.model,
                    logger=self.logger,
                ):
                    self.model.load_state_dict(self.early_stopper.best_state_dict)
                    self.logger.info("Loading best checkpoint and doing testing")
                    testing_metrics = self.eval(
                        train_graph=train_dataset, for_testing=True
                    )
                    self.logger.info(testing_metrics)

                    sys.exit(0)

    def test_time_aggregation(self) -> None:
        """
        TAG-CF specific function. Comparing the perforamnce before and after test-time aggregation
        """
        train_dataset = pre_process_graph(self.train_dataset)
        train_dataset = train_dataset.to(self.device)

        # Conducting evaluation without test-time aggregation
        self.logger.info("-----Performance without Test-time Aggregation-----")

        testing_metrics_before = self.eval(
            train_graph=train_dataset, for_testing=True, use_mp=False
        )
        self.logger.info("Testing results")
        self.logger.info(testing_metrics_before)

        # Conducting evaluation with test-time aggregation
        self.logger.info("-----Performance with Test-time Aggregation-----")
        if "m" in self.config and "n" in self.config:
            optimal_m = self.config["m"]
            optimal_n = self.config["n"]
        else:
            self.logger.info("-----m and n not specified. Conducting Grid Search-----")
            best_score, optimal_m, optimal_n = -torch.inf, 0, 0
            for m in TAGCF_SEARCH_SPACE:
                for n in TAGCF_SEARCH_SPACE:
                    self.model.message_passing_layer.m = m
                    self.model.message_passing_layer.n = n
                    eval_metrics = self.eval(
                        train_graph=train_dataset, for_testing=False, use_mp=True
                    )
                    if eval_metrics[self.config["early_stopping_metric"]] > best_score:
                        best_score, optimal_m, optimal_n = (
                            eval_metrics[self.config["early_stopping_metric"]],
                            m,
                            n,
                        )

        # Testing using TAG-CF with optimal m and n
        self.logger.info(f"Optimal m: {optimal_m}, Optimal n: {optimal_n}")
        self.model.message_passing_layer.m = optimal_m
        self.model.message_passing_layer.n = optimal_n
        testing_metrics_after = self.eval(
            train_graph=train_dataset, for_testing=True, use_mp=True
        )
        self.logger.info("Testing results")
        self.logger.info(testing_metrics_after)
        for metric in testing_metrics_after:
            self.logger.info(
                f"Improvement for {metric}: {(testing_metrics_after[metric] - testing_metrics_before[metric])*100/testing_metrics_before[metric]:.2f}%"
            )
