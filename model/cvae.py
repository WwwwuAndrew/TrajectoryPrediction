import torch
import torch.nn as nn
import utils.logger as logger

class CVAE(nn.Module):
    def __init__(self, inputDim, latentSize, outputDim):
        super().__init__()
        self.inputDim = inputDim
        self.latentSize = latentSize
        self.outputDim = outputDim

        self.encode = nn.Sequential()
        self.decode = nn.Sequential()

        self.linearMeans = nn.Linear(256, latentSize)
        self.linearLogVar = nn.Linear(256, latentSize)

        self.encode.add_module(name="E-Linear-1", module=nn.Linear(inputDim, 64))
        self.encode.add_module(name="E-ReLU-1", module=nn.ReLU())
        self.encode.add_module(name="E-Linear-2", module=nn.Linear(64, 128))
        self.encode.add_module(name="E-ReLU-2", module=nn.ReLU())
        self.encode.add_module(name="E-Linear-3", module=nn.Linear(128, 256))
        self.encode.add_module(name="E-ReLU-3", module=nn.ReLU())

        self.decode.add_module(name="D-Linear-1", module=nn.Linear(latentSize, 256))
        self.decode.add_module(name="D-ReLU-1", module=nn.Sigmoid())
        self.decode.add_module(name="D-Linear-2", module=nn.Linear(256, 128))
        self.decode.add_module(name="D-ReLU-2", module=nn.Sigmoid())
        self.decode.add_module(name="D-Linear-3", module=nn.Linear(128, outputDim))
        self.decode.add_module(name="D-ReLU-3", module=nn.ReLU())

    def forward(self, x):
        encodeResult = self.encode(x)
        means = self.linearMeans(encodeResult)
        logVars = self.linearLogVar(encodeResult)
        std = torch.exp(0.5*logVars)
        eps = torch.randn_like(std)
        z = means+eps*std
        g = self.decode(z)
        return g

