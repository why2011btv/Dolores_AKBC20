import os
import h5py
ent_dict = {}
rel_dict = {}
    
def build_dict(dictionary):
    if dictionary == ent_dict:
        path = "/home/why2011btv/research/OpenKE/benchmarks/FB15K237/entity2id.txt"
    else:
        path = "/home/why2011btv/research/OpenKE/benchmarks/FB15K237/relation2id.txt"
    with open(path,'r') as file:
        num = file.readline()

        for i in range(int(num)):
            cur_line = file.readline()
            tokens = cur_line.split('	')
            ID = int(tokens[1][:-1])
            cur_dict = {tokens[0]:ID}
            dictionary.update(cur_dict)

build_dict(ent_dict)
build_dict(rel_dict)
rev_ent_dict = {v:k for k,v in ent_dict.items()}  # {'001': 'a', '002': 'b'}
rev_rel_dict = {v:k for k,v in rel_dict.items()}  # {'001': 'a', '002': 'b'}

def look_up(input_ID):
    if input_ID <= 14540:
        ent_mid = rev_ent_dict[input_ID]
        return_val = ent_mid
        #print(return_val)
        cmd_str = 'grep -n ' + ent_mid + ' /home/why2011btv/Downloads/mid2name.txt'
        ent_name = os.popen(cmd_str)
        read_ent = ent_name.read()
        read_en = read_ent.split('	')
        try:
            name_ent = (read_en[1].split('\n'))[0]
        except:
            return_val = ent_mid
        else:


            #print(input_ID)
            #print(name_ent)
            a = name_ent.split(' ')
            m = ''
            for i in range(len(a)):
                if i < len(a)-1:
                    m = m + a[i] + '_'
                else:
                    m = m + a[i]
            #print(m)
            return_val = m
    elif input_ID <= 14777:
        
        input_ID -= 14541
        return_val = rev_rel_dict[input_ID]
        #print(return_val)
        
    else:
        print("ValueError!")
    return return_val
    
with h5py.File('./20180729.hdf5','r') as fin:
    a = fin['embedding'][...]
ent_emb = a[0:14541,:]
rel_emb = a[14541:14778,:]
with open('./20180730.txt','w') as fileout:
    line = '<14541> <100>\n'
    fileout.write(line)
    for i in range(14541):
        line = '<' + look_up(i) + '> <' + str(ent_emb[i].tolist()) + '>\n'
        fileout.write(line)