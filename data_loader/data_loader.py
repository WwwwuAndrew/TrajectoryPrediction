from .data_set import TrajectoryDataset, seqCollate
from torch.utils.data import DataLoader

def dataLoader(args, path):
    dataSet = TrajectoryDataset(
        path,
        obsLen=args.obs_len,
        predLen=args.pred_len,
        skip=args.skip,
        delim=args.delim
    )

    loader = DataLoader(
        dataSet,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=seqCollate
    )
    return dataSet, loader