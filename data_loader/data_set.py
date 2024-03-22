import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def read_file(file, delim):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(x) for x in line]
            data.append(line)
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    def __init__(self, dataDir, obsLen=8, predLen=12, skip=1, threshold=0.002, delim='\t'):
        super(TrajectoryDataset, self).__init__()
        self.dataDir = dataDir
        self.obsLen = obsLen
        self.predLen = predLen
        self.seqLen = self.obsLen + self.predLen
        self.skip = skip
        self.threshold = threshold
        self.delim = delim

        allFiles = os.listdir(self.dataDir)
        allFiles = [os.path.join(self.dataDir, _path) for _path in allFiles]

        for file in allFiles:
            data = read_file(file, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            framesData = []
            for frame in frames:
                framesData.append(data[frame == data[:, 0]])
            numSequences = int(math.ceil((len(frames)-self.seqLen+1)/self.skip))

            for idx in range(0, numSequences*self.skip+1, self.skip):
                curSeqData = np.concatenate(framesData[idx:idx+self.seqLen], axis=0)
                pedsInCurSeq = np.unique(curSeqData[:, 1])
                
        

    def __len__(slef):

    def __getitem__(self):