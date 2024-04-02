import json
import torch
import data_loader.data_loader as data_loader
import utils.losses as losses
import utils.logger as logger
import model.cvae as CVAE

def train():
    with open('config/config.json') as f:
        config = json.load(f)
    _, loader = data_loader.dataLoader(config, config['dataPath'])
    model = CVAE.CVAE(inputDim=8, latentSize=512, outputDim=12)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(config['epochs']):
        model.train()
        l2Loss = 0
        for _, data in enumerate(loader):
            obsTraj = data['obsTraj'].to(device)
            predTraj = data['predTraj'].to(device)
            fakeTraj = model(obsTraj)
            loss = losses.l2Loss(fakeTraj, predTraj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch: {} | Loss: {}'.format(epoch, loss.item()))


if __name__ == '__main__':
    train()