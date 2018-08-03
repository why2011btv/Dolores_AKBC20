import pickle
with open("/home/why2011btv/predicted_logit.txt",'rb') as file:
    bb = pickle.load(file)
    print(bb)
    
import h5py
import json
outfile = '/home/why2011btv/KG-embedding/20180727.hdf5'
with h5py.File(outfile, 'r') as fin:
    a = fin['embedding'][...]
ent_emb = a[0:14541,:]
rel_emb = a[14541:14778,:]

emb_dict = {}
emb_dict['ent_embeddings'] = ent_emb
emb_dict['rel_embeddings'] = rel_emb

ent_list = ent_emb.tolist()

rel_list = rel_emb.tolist()

emb_dict = {}
emb_dict['ent_embeddings'] = ent_list
emb_dict['rel_embeddings'] = rel_list

with open('/home/why2011btv/KG-embedding/my_emb.json', 'w') as emb_myjson:
    
    json.dump(emb_dict,emb_myjson)