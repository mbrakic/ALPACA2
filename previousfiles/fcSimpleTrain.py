import os
import warnings
from argparse import ArgumentParser
from LBDN.train import *
from LBDN.evaluate import *
# warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):

        self.seed = 123
        self.scale = 'small'
        self.layer = 'Sandwich'
        self.gamma = 1. 
        self.epochs = 100 

        self.mode = 'eval' 
        self.model = 'fcSimple' 
        self.dataset = 'cifar10' 
        self.loss = 'multimargin' 

        self.lr = 0.01 
        self.root_dir = 'LBDN/saved_models'
        self.train_batch_size = 256
        self.test_batch_size = 256 
        self.num_workers = 4 
        self.LLN = False 
        self.normalized = False
        self.cert_acc = False 

        self.lip_batch_size = 64
        self.print_freq = 10
        self.save_freq = 5
        
        self.in_channels = 3
        self.img_size = 32
        self.num_classes = 10

        self.width = {
            'small': 1,
            'medium': 2,
            'large': 4
        }[self.scale]

        if self.gamma is None:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}"
        elif self.LLN:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}-LLN-gamma{self.gamma:.1f}"
        else:
            self.train_dir = f"{self.root_dir}_seed{self.seed}/{self.dataset}/{self.model}-{self.layer}-{self.scale}-gamma{self.gamma:.1f}"

        os.makedirs("./data", exist_ok=True)
        os.makedirs(self.train_dir, exist_ok=True)

config = Config() 
# evaluate(config) 

train(config)