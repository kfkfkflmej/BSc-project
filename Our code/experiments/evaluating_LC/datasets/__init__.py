import pandas as pd
from omegaconf import DictConfig



def load_dataset(dataset_cfg: DictConfig):
    if dataset_cfg.name == "simple_qa" or dataset_cfg.name == "mini_simple_qa":
        return SimpleQADataset(dataset_cfg)
    elif dataset_cfg.name == "nq_open" or dataset_cfg.name == "mini_nq_open":
        return NQOPENDataset(dataset_cfg)
    elif dataset_cfg.name == "pop_qa" or dataset_cfg.name == "mini_pop_qa":
        return PopQADataset(dataset_cfg)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_cfg.name}")


class SimpleQADataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)
        

    def get_dataset(self):
        return self.df





class NQOPENDataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)
        

    def get_dataset(self):
        return self.df

    
    


class PopQADataset():
    def __init__(self, dataset_cfg: DictConfig):
        self.name = dataset_cfg.name
        self.dataset_cfg = dataset_cfg
        self.df = pd.read_csv(dataset_cfg.file_path)

    def get_dataset(self):
        return self.df

    