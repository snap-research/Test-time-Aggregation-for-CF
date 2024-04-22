from src.utils import load_yaml, pre_process_graph, init_logger
from src.model import ModelClass
from src.loss_function import LossFunction
from src.data import DatasetClass, get_dataloader
import dgl
from src.training_utils import EarlyStopper
import torch
from src.eval_metrics import MetricClass
from typing import Dict
import sys
from torch.utils.data import Dataset, DataLoader

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
    
        self.load_config(general_yaml_path=general_yaml_path,
                         model_yaml_path=model_yaml_path)
        
        self.load_dataset(self.config['dataset'])

        self.model = ModelClass[self.config['model_type']].value(
            number_of_users = self.dataset.n_user,
            number_of_items = self.dataset.n_item,
            embedding_dim = self.config['embedding_dim'],
            loss_function = LossFunction[self.config['loss_function']].value(),
            num_layers = self.config['n_layers'],
            embedding_table_weight = self.config.get("embedding_table_weight", None),
            m = self.config.get("m", None),
            n = self.config.get("n", None),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(lr=self.config['lr'],
                                          weight_decay=self.config['weight_decay'],
                                          params=self.model.parameters(),
        )

        self.early_stopper = EarlyStopper(
            early_stopping_patience=self.config['early_stopping_patience'],
            model_type = self.config["model_type"],
            loss_function = self.config["loss_function"],
            dataset = self.config["dataset"]
            )
        
        self.eval_metrics = [MetricClass[metric].value(top_k=k)  
                             for metric in self.config['metrics'] 
                                for k in self.config['top_k'] 
                                ]
        self.logger = init_logger()

    def load_config(self,
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
        self.device = self.config.get('device_id', 'cpu')
        self.config['n_layers'] = self.config.get('n_layers', 0)
        self.config.update(
            load_yaml(path=model_yaml_path)
            )
        
    def load_dataset(self,
                     dataset_name: str) -> None:
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

    def get_dataloader(self, 
                       dataset: dgl.DGLGraph) -> DataLoader:
        """
        get the dataloader given the graph

        Parameters
        ----------
        dataset: dgl.DGLGraph
            the dgl graph object that the dataloader gets data from
        """

        return get_dataloader(graph = dataset,
                              batch_size = self.config["batch_size"],
                              num_users = self.dataset.n_user,
                              num_items = self.dataset.n_item,
                              num_workers = self.config['num_workers'],
                              shuffle = True,
                              )
    
    def handle_iteration(self, 
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

    
    @ torch.no_grad()
    def eval(self, 
             train_graph: dgl.DGLGraph,
             use_mp: bool = False,
             for_testing: bool = False,
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
        all_embeddings = self.model.get_all_embbedings(graph=train_graph, 
                                                       use_mp=use_mp)
        
        user_embeddings = all_embeddings[:self.dataset.n_user]
        item_embeddings = all_embeddings[self.dataset.n_user:]

        # masking positive interactions observed during the training
        logits = torch.mm(user_embeddings, item_embeddings.t())
        users, items = self.train_dataset.edges()
        items = items - self.dataset.n_user
        # masking positive interactions duering the training 
        logits[users, items] = -9999
        # accordingly load evaluation labels
        labels = torch.zeros_like(logits)
        if for_testing:
            target_users, target_items = self.test_dataset.edges()
        else:
            target_users, target_items = self.valid_dataset.edges()
        target_items = target_items - self.dataset.n_user

        labels[target_users, target_items] = 1

        results = {}
        for metric in self.eval_metrics:
            metric_name = f"{metric.__class__.__name__}@{metric.top_k}"
            results[metric_name]  = metric(preds=logits,
                                           targets=labels)
        
        return results 

    def train(self):
        """
        the training logic for all matrix factorization models
        """
        train_dataloader = self.get_dataloader(
            dataset=self.train_dataset,
            )
        train_dataset = pre_process_graph(self.train_dataset)
        train_dataset = train_dataset.to(self.device)

        for epoch in range(self.config['total_epochs']):
            self.logger.info(f"Training on Epoch: {epoch}")
            for batch in train_dataloader:
                batch = batch.to(self.device)
                model_output, loss = self.model(
                    graph = train_dataset,
                    user_ids = batch[:, 0],
                    positive_item_ids = batch[:, 1],
                    negative_item_ids = batch[:, 2],
                    is_training = True,
                )
                self.handle_iteration(loss=loss)

            # conduct eval every eval_steps epochs  
            if (epoch + 1) % self.config["eval_steps"] == 0:
                eval_metrics = self.eval(train_graph=train_dataset, for_testing=False)
                self.logger.info(eval_metrics)
                if self.early_stopper(
                    current_score = eval_metrics[self.config["early_stopping_metric"]],
                    model = self.model,
                    logger = self.logger
                    ):
                    self.model.load_state_dict(self.early_stopper.best_state_dict)
                    self.logger.info("Loading best checkpoint and doing testing")
                    testing_metrics = self.eval(train_graph=train_dataset, for_testing=True)
                    self.logger.info(testing_metrics)
                    
                    sys.exit(0)

    def test(self):
        """
        TAG-CF specific function. Comparing the perforamnce before and after test-time aggregation 
        """
        train_dataset = pre_process_graph(self.train_dataset)
        train_dataset = train_dataset.to(self.device)
        self.logger.info("-----Performance without Test-time Aggregation-----")
        eval_metrics = self.eval(train_graph=train_dataset, for_testing=False, use_mp=False)
        self.logger.info("Validation results")
        self.logger.info(eval_metrics)

        testing_metrics = self.eval(train_graph=train_dataset, for_testing=True, use_mp=False)
        self.logger.info("Testing results")
        self.logger.info(testing_metrics)

        self.logger.info("-----Performance after Test-time Aggregation-----")
        eval_metrics = self.eval(train_graph=train_dataset, for_testing=False, use_mp=True)
        self.logger.info("Validation results")
        self.logger.info(eval_metrics)

        testing_metrics = self.eval(train_graph=train_dataset, for_testing=True, use_mp=True)
        self.logger.info("Testing results")
        self.logger.info(testing_metrics)