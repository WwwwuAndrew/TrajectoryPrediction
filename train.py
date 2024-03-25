import json
import torch
import data_loader.data_loader as data_loader
import util.losses as losses
import util.logger as logger

def train():
    with open('config.json') as f:
        config = json.load(f)