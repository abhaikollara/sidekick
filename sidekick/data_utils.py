import numpy as np

def train_test_split(datasets, split=0.2):
    train = []
    test = []
    # Ensure all datasets contain same no. of samples
    assert len(set(len(x) for x in datasets)) == 1
    split_point = int(len(datasets[0]) * (1-split))
    for dset in datasets:
        train.append(dset[:split_point])
        test.append(dset[split_point:])
    
    return {"train":train, "test":test}