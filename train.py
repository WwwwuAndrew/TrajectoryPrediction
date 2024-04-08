import json
import torch
import numpy as np
import data_loader.data_loader as data_loader
import utils.losses as losses
import utils.logger as logger
import model.cvae as CVAE

def train():
    with open('config/config.json') as f:
        config = json.load(f)
    dataSet, loader = data_loader.dataLoader(config, config['dataPath'])
    # dataSet, loader = data_loader.dataLoader(config, 'data_loader/test')
    model = CVAE.CVAE(inputDim=8, latentSize=512, outputDim=12)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(config['epochs']):
        model.train()
        for idx, data in enumerate(loader):
            obsTraj, predTraj, obsTrajRel, predTrajRel, nonLinearPed,\
            lossMask = data
            # logger.i("obsTraj->%s", np.array_repr(np.array(obsTraj)))
            # logger.i("predTraj->%s", np.array_repr(np.array(predTraj)))
            # logger.i("obsTrajRel->%s", np.array_repr(np.array(obsTrajRel)))
            # logger.i("predTrajRel->%s", np.array_repr(np.array(predTrajRel)))
            # logger.i("nonLinearPed->%s", np.array_repr(np.array(nonLinearPed)))
            # logger.i("lossMask->%s", np.array_repr(np.array(lossMask)))
            obsTraj = obsTraj.to(device)
            predTraj = predTraj.to(device)
            obsTrajRel = obsTrajRel.to(device)
            predTrajRel = predTrajRel.to(device)
            nonLinearPed = nonLinearPed.to(device)
            lossMask = lossMask.to(device)
            fakeTraj = model(obsTraj)
            loss = losses.l2Loss(fakeTraj, predTraj, mode='sum')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {} | Loss: {}'.format(epoch, loss.item()))


if __name__ == '__main__':
    train()