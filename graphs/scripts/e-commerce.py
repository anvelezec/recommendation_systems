import kagglehub
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np 
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sentence_transformers import SentenceTransformer

# Download latest version
path = kagglehub.dataset_download("mervemenekse/ecommerce-dataset")

print("Path to dataset files:", path)

data = pd.read_csv(path + "/E-commerce Dataset.csv")
print(data.head(5))

products = data[["Product_Category", "Product"]].drop_duplicates()
transactions = data[["Customer_Id", "Gender", "Device_Type", "Customer_Login_type"]]


def load_node(dataset, node_col, encoders=None, **kwargs):
    # Extract {value, index} pairs
    x = None
    mapping = {}
    for i, index in enumerate(dataset[node_col].unique()):
        if isinstance(index, np.int64):
            index = int(index)
        mapping.update({index: i})
    if encoders is not None:
        features = [encoder(dataset[col]) for col, encoder in encoders.items()]
        x = torch.cat(features, dim=-1)

    return x, mapping

def load_edge(dataset, 
              source_index_col, 
              source_mapping, 
              destiniy_index_col, 
              destiny_mapping, 
              encoders=None, **kwargs):
    
    source = [source_mapping[value] for value in dataset[source_index_col]]
    destiny = [destiny_mapping[value] for value in dataset[destiniy_index_col]]
    edge_index = torch.tensor([source, destiny])

    edge_features = None
    if encoders is not None:
        features = [encoder(dataset[col]) for col, encoder in encoders.items()]
        edge_features = torch.cat(features, dim=-1)

    return edge_features, edge_index


class SequenceEncoder:
    # convert text to embeddings
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
    
    @torch.no_grad()
    def __call__(self, df, *args, **kwds):
        x = self.model.encode(df.values, show_process_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()
        


# extracting nodes features and mappings
customer_features, customer_mapping = load_node(transactions, "Customer_Id", encoders=None)
product_features, product_mapping = load_node(products, "Product", encoders={"Product_Category": SequenceEncoder()})


# extracts edges features and mappings
edge_features, edge_index = load_edge(data, "Customer_Id", customer_mapping, 
                                      "Product", product_mapping, encoders=None)

data = HeteroData()
data["customer"].num_nodes = len(customer_mapping)
data["product"].num_nodes = len(product_mapping)
data["product"].x = product_features
data["customer", "buy", "product"].edge_index = edge_index
data["customer", "buy", "product"].edge_label = edge_features

# customer 38997
# product 42
# edge index 51290
print(data)

# * Symmetry: Many GNNs operate symmetrically, so converting the graph to undirected ensures all connections are bidirectional.
# * Message Passing: In undirected graphs, information flows in both directions. This can improve the learning process for node and edge embeddings.
# * Simplification: Some graph tasks don't distinguish between the direction of edges (e.g., community detection), making undirected graphs more appropriate.

data = ToUndirected()(data)
del data["customer", "rev_buy", "product"].edge_label

