from src.utils import load_yaml, pre_process_graph
from src.model import ModelClass
from src.loss_function import LossFunction
from src.data import DatasetClass, get_dataloader
import dgl
from src.training_utils import EarlyStopper

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
        self.early_stopper = EarlyStopper(
            early_stopping_patience=self.config['early_stopping_patience']
            )
        

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
                              num_workers = self.config['num_workers'],
                              shuffle = True,
                              )
    
    def train(self):
        train_dataloader = self.get_dataloader(
            dataset=self.train_dataset,
            )
        self.train_dataset = pre_process_graph(self.train_dataset)
        self.train_dataset = self.train_dataset.to(self.device)

        for epoch in self.config['total_epochs']:
            for batch in train_dataloader:
                model_output, loss = self.model(graph=self.train_dataset,
                                                )
        return next(iter(train_dataloader))