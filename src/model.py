import torch
from src.loss_function import BaseLossFunction
from abc import ABC, abstractmethod
import dgl
from dgl.utils import expand_as_pair
from src.constants import DEFAULT_NODE_FEATURE_FIELD, DEFAULT_NODE_INDICES_FIELD
from src.loss_function import LossFunction
from dgl import function as fn
from src.data import TripletModelOutput
from typing import Tuple, Optional, Union
from enum import Enum

class MessagePassingLayer(torch.nn.Module):

    """ Non-Paramtrized Message Passing Layer
    
    Parameters
    ----------
    m: float
        the in-degree normalization factor for the message passing layer
    n: float
        the out-degree normalization factor for the message passing layer
    """
    def __init__(self, 
                 m: float = - 0.5,
                 n: float = - 0.5,
                 ) -> None:
        
        super().__init__()
        self.m = m
        self.n = n
        self.aggregate_fn = fn.copy_u('h', 'm')

    def forward(self, 
                graph: dgl.DGLGraph,
                ) -> torch.Tensor:
        
        feat = graph.ndata[DEFAULT_NODE_FEATURE_FIELD]
        
        # Implementation based on 
        # https://docs.dgl.ai/en/0.9.x/_modules/dgl/nn/pytorch/conv/graphconv.html#GraphConv
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            # Conducting out-degree normalization over node features 
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, self.m)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            # Message Aggregation
            graph.srcdata['h'] = feat_src
            graph.update_all(self.aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            # Conducting in-degree normalization over aggregated features
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, self.n)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        return rst

class BaseMatrixFactorization(ABC, torch.nn.Module):
    """ Base Matrix Facotrization Model

    Parameters
    ----------
    number_of_users: int
        number of users in the dataset
    number_of_items: int
        number of items in the dataset
    embedding_dim: int
        embedding dimension
    loss_function: BaseLossFunction
        loss function to be used in the model
    """

    def __init__(self, 
                 number_of_users: int,
                 number_of_items: int,
                 embedding_dim: int,
                 loss_function: BaseLossFunction = None, 
                 ) -> None:
    
        super().__init__()

        self.embedding_table = torch.nn.Embedding(
            number_of_users+number_of_items, 
            embedding_dim)
        self.number_of_users = number_of_users
        self.loss_function = loss_function

    def get_user_embedding(self, 
                           user_ids: torch.Tensor
                           ) -> torch.Tensor:
        
        return self.embedding_table(
            user_ids
            )
    

    def get_item_embedding(self, 
                           item_ids: torch.Tensor
                           ) -> torch.Tensor:
        
        # we assumethat item_ids are offset by the number of users
        return self.embedding_table(
            item_ids
            )
    
    @abstractmethod
    def forward(self, 
                **kwargs,
                ) -> Tuple[TripletModelOutput, torch.Tensor]:
        
        raise NotImplementedError("Must be implemented in child classes")
    
    @abstractmethod
    def get_all_embbedings(self, 
                **kwargs,
                ) -> torch.Tensor:
        
        raise NotImplementedError("Must be implemented in child classes")
    
class MatrixFactorization(BaseMatrixFactorization):
    """ Matrix Factorization Model

    Parameters
    ----------
    number_of_users: int
        number of users in the dataset
    number_of_items: int
        number of items in the dataset
    embedding_dim: int
        embedding dimension
    loss_function: BaseLossFunction
        loss function to be used in the model
    """

    def __init__(self, 
                 number_of_users: int,
                 number_of_items: int,
                 embedding_dim: int,
                 loss_function: BaseLossFunction, 
                 **kwargs,
                 ) -> None:
    
        super().__init__(number_of_users=number_of_users,
                         number_of_items=number_of_items,
                         embedding_dim=embedding_dim,
                         loss_function=loss_function)

    def forward(self, 
                graph: dgl.DGLGraph,
                user_ids: Optional[torch.Tensor] = None,
                positive_item_ids: Optional[torch.Tensor] = None,
                negative_item_ids: Optional[torch.Tensor] = None,
                is_training = False,
                ) -> Union[TripletModelOutput, Tuple[TripletModelOutput, torch.Tensor]]:
        
        user_embedding = self.get_user_embedding(user_ids)
        positive_item_embedding = self.get_item_embedding(positive_item_ids)
        
        if negative_item_ids is not None:
            negative_item_embedding = self.get_item_embedding(negative_item_ids)
        else:
            negative_item_embedding = None
        
        model_output = TripletModelOutput(user_embedding = user_embedding, 
                                           positive_item_embedding = positive_item_embedding, 
                                           negative_item_embedding= negative_item_embedding)
                                          
        if is_training:                                          
            return model_output, self.loss_function.get_loss(model_output)
        else:
            return model_output
    
    def get_all_embbedings(self, **kwargs) -> torch.Tensor:
        return self.embedding_table.weight

class LightGCN(BaseMatrixFactorization):
    """ LightGCN Model

    Parameters
    ----------
    number_of_users: int
        number of users in the dataset
    number_of_items: int
        number of items in the dataset
    embedding_dim: int
        embedding dimension
    num_layers: int
        number of message passing layers in the model
    loss_function: BaseLossFunction
        loss function to be used in the model, default to BPR
    """
    def __init__(self, 
                 number_of_users: int, 
                 number_of_items: int, 
                 embedding_dim: int, 
                 num_layers: int,
                 loss_function: BaseLossFunction = \
                    LossFunction['BPR'].value(),
                 **kwargs,
                ) -> None:
        
        super().__init__(number_of_users = number_of_users, 
                         number_of_items = number_of_items, 
                         embedding_dim = embedding_dim, 
                         loss_function = loss_function)
        
        self.num_layers = num_layers
        self.message_passing_layer = MessagePassingLayer()

    def forward(self, 
                graph: dgl.DGLGraph,
                user_ids: Optional[torch.Tensor] = None,
                positive_item_ids: Optional[torch.Tensor] = None,
                negative_item_ids: Optional[torch.Tensor] = None,
                is_training = False,
                ) -> Union[TripletModelOutput, Tuple[TripletModelOutput, torch.Tensor]]:
        
        total_embedding = self.message_passing(graph=graph)
        user_embedding = total_embedding[user_ids] if user_ids != None else None
        positive_item_embedding = total_embedding[positive_item_ids] if positive_item_ids != None else None
        negative_item_embedding = total_embedding[negative_item_ids] if negative_item_ids != None else None

        model_output = TripletModelOutput(user_embedding = user_embedding, 
                                           positive_item_embedding = positive_item_embedding, 
                                           negative_item_embedding= negative_item_embedding)

        if is_training:                                          
            return model_output, self.loss_function.get_loss(model_output)
        else:
            return model_output

    
    def message_passing(self, 
                graph: dgl.DGLGraph,
                ) -> torch.Tensor:
        
        node_indices = graph.ndata[DEFAULT_NODE_INDICES_FIELD]
        graph.ndata[DEFAULT_NODE_FEATURE_FIELD] = self.embedding_table(node_indices)

        results = [graph.ndata[DEFAULT_NODE_FEATURE_FIELD]]

        for _ in range(self.num_layers):
            intermediate_results = self.message_passing_layer(graph)
            graph.ndata[DEFAULT_NODE_FEATURE_FIELD] = intermediate_results
            results.append(intermediate_results)
        
        return torch.cat(results, dim=1).mean(1)
    
    def get_all_embbedings(self, 
                           graph= dgl.DGLGraph,
                           **kwargs) -> torch.Tensor:
        return self.message_passing(graph=graph)

class TAGCF(LightGCN):
    """ TAG-CF Model
    
    Parameters
    ----------
    number_of_users: int
        number of users in the dataset
    number_of_items: int
        number of items in the dataset
    embedding_dim: int
        embedding dimension
    embedding_table_weight: torch.Tensor
        pre-trained embedding table weight
    m: float
        the in-degree normalization factor for the message passing layer
    n: float
        the out-degree normalization factor for the message passing layer
    """
    def __init__(self, 
                 number_of_users: int, 
                 number_of_items: int, 
                 embedding_dim: int, 
                 num_layers: int,
                 embedding_table_weight: str,
                 m: float,
                 n: float,
                 **kwargs
                 ) -> None:
        
        super().__init__(number_of_users = number_of_users, 
                         number_of_items = number_of_items, 
                         embedding_dim = embedding_dim,
                         num_layers = num_layers,
                         loss_function=None, # TAG-CF is a test-time augmentation framework, no loss required
                         )

        embedding_table_weight = torch.load(
            embedding_table_weight, 
            map_location='cpu',
            )['embedding_table.weight']
        self.embedding_table.weight = torch.nn.Parameter(embedding_table_weight)
        self.message_passing_layer.m = m
        self.message_passing_layer.n = n

    @torch.no_grad()
    def forward(self, 
                graph: dgl.DGLGraph,
                user_indices: Optional[torch.Tensor] = None,
                positive_item_indices: Optional[torch.Tensor] = None,
                negative_item_indices: Optional[torch.Tensor] = None,
                ) -> TripletModelOutput:
        
        model_output = super().forward(graph=graph,
                                       user_indices=user_indices,
                                       positive_item_indices=positive_item_indices,
                                       negative_item_indices=negative_item_indices,
                                       is_training=False)
        return model_output
    
    @torch.no_grad()
    def get_all_embbedings(self, 
                           graph: dgl.DGLGraph,
                           use_mp: bool,
                           **kwargs) -> torch.Tensor:
        
        if use_mp:
            return self.message_passing(graph=graph)
        else:
            return self.embedding_table.weight



class ModelClass(Enum):
    MF = MatrixFactorization
    LGCN = LightGCN
    TAGCF = TAGCF