import json
import torch
import data_loader.data_loader as data_loader
import util.losses as losses
import util.logger as logger
import data_loader.data_set as data_set

def train():
    with open('config.json') as f:
        config = json.load(f)
    dset = data_set.TrajectoryDataset('data_loader/test',
        obsLen=1,
        predLen=2,
        skip=1,
        delim='\t')
    print(dset.__getitem__(1))

if __name__ == '__main__':
    train()