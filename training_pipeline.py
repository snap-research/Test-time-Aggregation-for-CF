from src.utils import load_yaml, pre_process_graph
from src.model import ModelClass
from src.loss_function import LossFunction
from src.data import DatasetClass, get_dataloader
import dgl
from src.training_utils import EarlyStopper
import torch
from tqdm import tqdm 
import numpy as np
from src.eval_metrics import MetricClass
from typing import Dict


class MFPipeline:

    def __init__(
        self,
        yaml_path: str,
    ):
    
        self.load_config(yaml_path=yaml_path)
        self.load_dataset(self.config['dataset'])

        self.model = ModelClass[self.config['model_type']].value(
            number_of_users = self.dataset.n_user,
            number_of_items = self.dataset.n_item,
            embedding_dim = self.config['embedding_dim'],
            loss_function = LossFunction[self.config['loss_function']].value(),
            num_layers = self.config['n_layers'],
        )

        self.optimizer = torch.optim.Adam(lr=self.config['lr'],
                                          weight_decay=self.config['weight_decay'],
                                          params=self.model.parameters(),
        )

        self.early_stopper = EarlyStopper(
            early_stopping_patience=self.config['early_stopping_patience']
            )
        
        self.eval_metrics = [MetricClass[metric].value(top_k=k)  
                             for metric in self.config['metrics'] 
                                for k in self.config['top_k'] 
                                ]

    def load_config(self,
                    yaml_path: str
                    ) -> None:
        self.config = load_yaml(path=yaml_path)
        self.device = self.config.get('device_id', 'cpu')

    def load_dataset(self,
                     dataset_name) -> None:

        self.dataset = DatasetClass[dataset_name].value()
        self.train_dataset = self.dataset.get_train()
        self.valid_dataset = self.dataset.get_valid()
        self.test_dataset = self.dataset.get_test()

    def get_dataloader(self, 
                       dataset: dgl.DGLGraph):

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
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


    @ torch.no_grad()
    def eval(self, 
             train_graph: dgl.DGLGraph,
             for_testing: bool = False,
             ) -> Dict[str, float]:
        
        # acquiring all embeddings for users and items 
        all_embeddings = self.model.get_all_embbedings(graph=train_graph)
        user_embeddings = all_embeddings[:self.dataset.n_user]
        item_embeddings = all_embeddings[self.dataset.n_user:]

        # masking positive interactions observed during the training
        logits = torch.mm(user_embeddings, item_embeddings.t())
        users, items = self.train_dataset.edges()
        logits[users, items] = -np.inf
        labels = torch.zeros_like(logits)
        if for_testing:
            target_users, target_items = self.test_dataset.edges()
        else:
            target_users, target_items = self.valid_dataset.edges()

        labels[target_users, target_items] = 1
        
        results = {}
        for metric in self.eval_metrics:
            metric_name = f"{metric.__class__}@{metric.top_k}"
            results[metric_name]  = metric(preds=logits,
                                           target=labels).item()
        
        return results 

    def train(self):
        train_dataloader = self.get_dataloader(
            dataset=self.train_dataset,
            )
        train_dataset = pre_process_graph(self.train_dataset)
        train_dataset = self.train_dataset.to(self.device)

        for epoch in range(self.config['total_epochs']):
            for batch in train_dataloader:
                batch = batch.to(self.device)
                model_output, loss = self.model(graph = train_dataset,
                                                user_ids = batch[:, 0],
                                                positive_item_ids = batch[:, 1],
                                                negative_item_ids = batch[:, 2],
                                                is_training = True,
                                                )
                self.handle_iteration(loss=loss)

            if (epoch + 1) % self.config["eval_steps"]:
                print(self.eval(train_graph=train_dataset, for_testing=False))