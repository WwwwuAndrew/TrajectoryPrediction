import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.logger as logger

def seqCollate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    # obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    # pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    # obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    # pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    # non_linear_ped = torch.cat(non_linear_ped_list)
    # loss_mask = torch.cat(loss_mask_list, dim=0)
    obs_traj = torch.tensor(obs_seq_list)
    pred_traj = torch.tensor(pred_seq_list)
    obs_traj_rel = torch.tensor(obs_seq_rel_list)
    pred_traj_rel = torch.tensor(pred_seq_rel_list)
    non_linear_ped = torch.tensor(non_linear_ped_list)
    loss_mask = torch.tensor(loss_mask_list)
    seq_start_end = torch.LongTensor(seq_start_end)
    # out = [
    #     obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
    #     loss_mask, seq_start_end
    # ]
    out = [
        obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
        non_linear_ped_list, loss_mask_list, seq_start_end
    ]

    return tuple(out)

def readFile(file, delim):
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

def polyFit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    def __init__(self, dataDir, obsLen=8, predLen=12, skip=1, threshold=0.002, minPed=1, delim='\t'):
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
        numPedsInSeq = []
        seqList = []
        seqListRel = []
        lossMaskList = []
        nonLinearPed = []

        for file in allFiles:
            logger.i("file->%s", file)
            data = readFile(file, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            # logger.i('frames-> %s', np.array_repr(np.array(frames)))
            framesData = []
            for frame in frames:
                framesData.append(data[frame == data[:, 0]])
                # logger.i('append framesData-> %s', np.array_repr(np.array(framesData)))
            numSequences = int(math.ceil((len(frames)-self.seqLen+1)/self.skip))
            # logger.i('numSequences-> %d', numSequences)

            for idx in range(0, numSequences*self.skip+1, self.skip):
                currSeqData = np.concatenate(framesData[idx:idx+self.seqLen], axis=0)
                # logger.i('append currSeqData-> %s', np.array_repr(np.array(currSeqData)))
                pedsInCurSeq = np.unique(currSeqData[:, 1])
                # logger.i('append pedsInCurSeq-> %s', np.array_repr(np.array(pedsInCurSeq)))
                currSeqRel = np.zeros((len(pedsInCurSeq), 2, self.seqLen))
                currSeq = np.zeros((len(pedsInCurSeq), 2, self.seqLen,))
                currLossMask = np.zeros((len(pedsInCurSeq), self.seqLen))
                numPedsConsidered = 0
                nonLinearPed = []

                for _, pedId in enumerate(pedsInCurSeq):
                    # logger.i("pedId = %d", pedId)
                    currPedSeq = currSeqData[currSeqData[:, 1] == pedId, :]
                    currPedSeq = np.around(currPedSeq, decimals=4)
                    # logger.i('append currPedSeq-> %s', np.array_repr(np.array(currPedSeq)))
                    padFront = frames.index(currPedSeq[0, 0]) - idx
                    padEnd = frames.index(currPedSeq[-1, 0]) - idx + 1
                    # logger.i("padFront = %d , padEnd = %d", padFront, padEnd)
                    if padEnd-padFront != self.seqLen:
                        continue
                    currPedSeq = np.transpose(currPedSeq[:, 2:])
                    currPedSeq = currPedSeq
                    # logger.i('append currPedSeq-> %s', np.array_repr(np.array(currPedSeq)))
                    relCurrPedSeq = np.zeros(currPedSeq.shape)
                    relCurrPedSeq[:, 1:] = currPedSeq[:, 1:]-currPedSeq[:, :-1]
                    # logger.i('append relCurrPedSeq-> %s', np.array_repr(np.array(relCurrPedSeq)))
                    _idx = numPedsConsidered
                    currSeq[_idx, :, padFront:padEnd] = currPedSeq
                    # logger.i('append currSeq-> %s', np.array_repr(np.array(currSeq)))
                    currSeqRel[_idx, :, padFront:padEnd] = relCurrPedSeq
                    # logger.i('append currSeqRel-> %s', np.array_repr(np.array(currSeqRel)))
                    nonLinearPed.append(polyFit(currPedSeq, predLen, threshold))
                    # logger.i('append nonLinearPed-> %s', np.array_repr(np.array(nonLinearPed)))
                    currLossMask[_idx, padFront:padEnd] = 1
                    # logger.i('append currLossMask-> %s', np.array_repr(np.array(currLossMask)))
                    numPedsConsidered += 1
                
                if numPedsConsidered > minPed:        
                    nonLinearPed += nonLinearPed
                    numPedsInSeq.append(numPedsConsidered)
                    lossMaskList.append(currLossMask[:numPedsConsidered])
                    seqList.append(currSeq[:numPedsConsidered])
                    seqListRel.append(currSeqRel[:numPedsConsidered])
        
        self.numSeq = len(seqList)
        seqList = np.concatenate(seqList, axis=0)
        seqListRel = np.concatenate(seqListRel, axis=0)
        lossMaskList = np.concatenate(lossMaskList, axis=0)
        nonLinearPed = np.asarray(nonLinearPed)

        self.obsTraj = torch.from_numpy(seqList[:, :, :self.obsLen]).type(torch.float)
        self.predTraj = torch.from_numpy(seqList[:, :, self.obsLen:]).type(torch.float)
        self.obsTrajRel = torch.from_numpy(seqListRel[:, :, :self.obsLen]).type(torch.float)
        self.predTrajRel = torch.from_numpy(seqListRel[:, :, self.obsLen:]).type(torch.float)
        self.lossMask = torch.from_numpy(lossMaskList).type(torch.float)
        self.nonLinearPed = torch.from_numpy(nonLinearPed).type(torch.float)
        cumStartIdx = [0] + np.cumsum(numPedsInSeq).tolist()
        self.seqStartEnd = [
            (start, end)
            for start, end in zip(cumStartIdx, cumStartIdx[1:])
        ]

    def __len__(self):
        return self.numSeq

    def __getitem__(self, index):
        start, end = self.seqStartEnd[index]
        out = [
            self.obsTraj[start:end, :], self.predTraj[start:end, :],
            self.obsTrajRel[start:end, :], self.predTrajRel[start:end, :],
            self.nonLinearPed[start:end], self.lossMask[start:end, :]
        ]
        return out