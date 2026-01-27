import os
import pickle as pkl

def read_halomaker(path: str, iout:int | None=None):
    raise NotImplementedError

def read_ptree(path: str, iout:int | None=None):
    if iout is not None:
        path = os.path.join(path, f"ptree_{iout:03d}.pkl")
    return pkl.load(open(path, "rb"))