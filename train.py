import json
import torch
import data_loader.data_loader as data_loader
import utils.losses as losses
import utils.logger as logger
import data_loader.data_set as data_set
import model.cvae as CVAE

def train():
    with open('config/config.json') as f:
        config = json.load(f)
    dset = data_set.TrajectoryDataset('data_sets/ETH-UCY/eth/train',
        obsLen= config['obsLen'],
        predLen=config['predLen'],
        skip=   config['skip'],
        delim=  config['delim'])
    print(dset.__getitem__(1))


if __name__ == '__main__':
    train()