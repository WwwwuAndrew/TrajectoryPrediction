from .data_set import TrajectoryDataset, seqCollate
from torch.utils.data import DataLoader

def myCollateFn(batch):
    return batch[0]

def dataLoader(args, path):
    dataSet = TrajectoryDataset(
        path,
        obsLen  = args['obsLen'],
        predLen = args['predLen'],
        skip    = args['skip'],
        delim   = args['delim']
    )

    loader = DataLoader(
        dataSet,
        batch_size  = args['batchSize'],
        shuffle     = True,
        num_workers = args['loaderNumWorkers'],
        collate_fn  = myCollateFn
    )
    return dataSet, loader