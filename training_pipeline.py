from src.utils import load_yaml
from src.model import ModelClass
from src.loss_function import LossFunction
from src.data import DatasetClass


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

    def load_config(self,
                    yaml_path: str
                    ) -> None:
        self.config = load_yaml(path=yaml_path)

    def load_dataset(self,
                     dataset_name) -> None:

        self.dataset = DatasetClass[dataset_name].value()
        self.train_dataset = self.dataset.get_train()
        self.valid_dataset = self.dataset.get_valid()
        self.test_dataset = self.dataset.get_test()


