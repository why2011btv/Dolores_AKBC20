import numpy as np
import argparse
import pickle
from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
            token_ids_reverse = np.zeros([batch_size, num_steps], np.int32)
            next_token_id_reverse = np.zeros([batch_size, num_steps], np.int32)
            
            token_ids = self.paths_repre_by_IDs[i:i+batch_size,0:-1]    # [batch_size, num_steps] (model.token_ids)
            next_token_id[:,0:-1] = self.paths_repre_by_IDs[i:i+batch_size,2:]
            next_token_id[:,-1:] = np.full((batch_size,1), 14778)
            
            token_ids_reverse = np.flip(self.paths_repre_by_IDs[i:i+batch_size,1:],1)
            next_token_id_reverse[:,0:-1] = np.flip(self.paths_repre_by_IDs[i:i+batch_size,:-2],1)
            next_token_id_reverse[:,-1:] = np.full((batch_size,1), 14778)
            i += batch_size
            X = {'token_ids': token_ids,
                 'next_token_id': next_token_id,
                 'token_ids_reverse': token_ids_reverse,
                 'next_token_id_reverse': next_token_id_reverse}
            yield X


def main(args):
    ent_num = 14541
    with open("/home/why2011btv/research/OpenKE/benchmarks/FB15K237/test2id.txt",'r') as f:
    #with open("/home/why2011btv/KG-embedding/obama.txt",'r') as f:
        lines = f.readlines()
        triplet_num = len(lines)-1
        print("triplet_num:",triplet_num)
        #triplet_num = 600
        test_set = np.zeros([triplet_num, 3],np.int32)
        i = 0
        for line in lines:
            a = line.split(' ')
            if len(a) > 1 and i<triplet_num:
                #a[2] = (a[2])[:-1] #because of newline
                #a[1] = a[1][:-1]
                
#            test_set[i][0] = int(a[0])
#            test_set[i][1] = int(a[2])
#            test_set[i][2] = int(a[1])
                aa = 1
                test_set[i][0] = int(a[0])
                test_set[i][1] = int(a[2]) + ent_num
                test_set[i][2] = int(a[1])
                #print("a[0]:",test_set[i][0])
                #print("a[2]:",test_set[i][1])
                #print("a[1]:",test_set[i][2])
                #print("a:",aa)
                #print(test_set)
                i += 1

    
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)
    data = MYDataset(test_set)
    
    perplexity = test(options, ckpt_file, data, batch_size=2)
    return perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    args = parser.parse_args()
    main(args)

