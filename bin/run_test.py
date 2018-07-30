import numpy as np
import argparse
import pickle
from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

class MYDataset(object):
    def __init__(self, paths_repre_by_IDs):
        self.paths_repre_by_IDs = paths_repre_by_IDs
    def iter_batches(self, batch_size, num_steps):
        i = 0
        while i <= self.paths_repre_by_IDs.shape[0]:
            if (i + batch_size) > self.paths_repre_by_IDs.shape[0]:
                break
            token_ids = np.zeros([batch_size, num_steps], np.int32)
            next_token_id = np.zeros([batch_size, num_steps], np.int32)
            
            token_ids = self.paths_repre_by_IDs[i:i+batch_size,:]    # [batch_size, num_steps] (model.token_ids)
            next_token_id[:,0:-1] = token_ids[:,1:]
            next_token_id[:,-1:] = np.full((batch_size,1), 16296)
            i += batch_size
            X = {'token_ids': token_ids,
                 'next_token_id': next_token_id}
            yield X


def main():
    test_set = np.zeros([200, 3],np.int32)
    with open("/home/why2011btv/KG-embedding/train2id.txt",'r') as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            a = line.split(' ')
#            a[2] = (a[2])[:-1] #because of newline
            
#            test_set[i][0] = int(a[0])
#            test_set[i][1] = int(a[2])
#            test_set[i][2] = int(a[1])
            print("a[0]:",a[0])
            print("a[2]:",a[2])
            print("a[1]:",a[1])
            test_set[i][0] = a[0]
            test_set[i][1] = a[2]
            test_set[i][2] = a[1]
            i += 1

    
    options, ckpt_file = load_options_latest_checkpoint("/home/why2011btv/KG-embedding/20180727/")
    data = MYDataset(test_set)
    
    test(options, ckpt_file, data, batch_size=6)


if __name__ == '__main__':
    main()

