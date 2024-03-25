import torch

def l2Loss(predTraj, predTrajGt, lossMask, random=0, mode='average'):
    seqLen, batch, _ = predTraj.size()
    loss = (lossMask.unsqueeze(dim=2) *
            (predTrajGt.permute(1, 0, 2) - predTraj.permute(1, 0, 2))**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(lossMask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)


def displacementError(predTraj, predTrajGt, considerPed=None, mode='sum'):
    seqLen, _, _ = predTraj.size()
    loss = predTrajGt.permute(1, 0, 2) - predTraj.permute(1, 0, 2)
    loss = loss**2
    if considerPed is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * considerPed
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def finalDisplacementError(pred_pos, pred_posGt, considerPed=None, mode='sum'):
    loss = pred_posGt - pred_pos
    loss = loss**2
    if considerPed is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * considerPed
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)