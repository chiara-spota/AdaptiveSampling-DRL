import yaml
import torch

class Config():

    """Object to hold the config requirements for an agent"""
    def __init__(self, args):
        with open(args, 'r') as f:
            yaml_configs = yaml.safe_load(f)
        for arg_name, arg_value in yaml_configs.items():
            self.__dict__[arg_name] = arg_value

        if self.__dict__['use_gpu']:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'

        self.num_to_action = None