
'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import bilm.node2vec as node2vec
from gensim.models import Word2Vec
from datetime import datetime
import pickle
import tensorflow as tf

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--relation', dest='relationed', action='store_true',
						help='Boolean specifying (un)relationed. Default is unrelationed.')
	parser.add_argument('--unrelation', dest='unrelationed', action='store_false')
	parser.set_defaults(relationed=False)		

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	weighted = 0
	relationed = 1
	graph_input = "/home/why2011btv/research/node2vec/graph/train2id.edgelist"
	directed = 0
	if weighted:
		G = nx.read_edgelist(graph_input, nodetype=str, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		if relationed:
			G = nx.read_edgelist(graph_input, nodetype=str, data=(('relation',str),), create_using=nx.DiGraph())
			for edge in G.edges():
				G[edge[0]][edge[1]]['weight'] = 1
		else:
			G = nx.read_edgelist(graph_input, nodetype=str, create_using=nx.DiGraph())
			for edge in G.edges():
				G[edge[0]][edge[1]]['weight'] = 1

	if not directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.save_word2vec_format(args.output)
	
	return

def calcu_METRIC(logits):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    ent_num = 14541
    with open("/home/why2011btv/research/OpenKE/benchmarks/FB15K237/test2id.txt",'r') as f:
        lines = f.readlines()
        triplet_num = len(lines)-1
        test_set = np.zeros([triplet_num, 3],np.int32)
        i = 0
        for line in lines:
            a = line.split(' ')
            if len(a) > 1 and i<triplet_num:
                
                #a[2] = (a[2])[:-1] #because of newline
                #test_set[i][0] = int(a[0])
                #test_set[i][1] = int(a[2])
                #test_set[i][2] = int(a[1])
                test_set[i][0] = int(a[0])
                test_set[i][1] = int(a[2]) + ent_num
                test_set[i][2] = int(a[1])
                #print(test_set)
                i += 1
    
    print(str(datetime.now()))
    #file_output = open("./output201807122130.txt",'wb')
    nx_G = read_graph()
    directed = 0
    p = 1
    q = 1
    G = node2vec.Graph(nx_G, directed, p, q)
    print("\nGraph established!")
    #aaa = G.G['/m/027rn']['/m/06cx9']["relation"]
    #print(aaa)
    #print(nx_G.neighbors('14514'))
    #print(nx_G.number_of_nodes())
    
    
    l_rank_list = []
    l_rank_recipr_list = []
    l_rank_list_filter = []
    l_rank_recipr_list_filter = []
    r_rank_list = []
    r_rank_recipr_list = []
    r_rank_list_filter = []
    r_rank_recipr_list_filter = []

    def calculate_metrics(L_or_R, rank_list, rank_recipr_list, rank_list_filter, rank_recipr_list_filter):
        if L_or_R == 0:    # R
            head = 0
            tail = 2
        else:              # L
            head = 2
            tail = 0
    
    
        with open("/home/why2011btv/2018-08-02_21-00-02predicted_logit.txt", 'rb') as file_logit:
            logit_s = pickle.load(file_logit)
            # logits is a parameter from main function
            batch_size = (logits[0][0].shape)[0]
            batch_num = len(logits)
            print(batch_size)
            print(batch_num)
            #batch_num = 100
            for i in range(batch_num):
                for j in range(batch_size):
                    cur_dict = {}
                    cur_dict_filter = {}
                    cur_logit = logits[i][L_or_R][j]
                    for k in range(ent_num):
                        cur_dict[k] = cur_logit[k]
                    cur_dict_filter.update(cur_dict)
                    cur_node = test_set[batch_size*i + j][head]
                    try:
                        cur_list = nx_G.neighbors(str(cur_node))
                    except:
                        print("current node %d does not exist!" % cur_node)
                    else:
                        flag_exclude = 1
                        if flag_exclude:
                            for neighb in cur_list:
                                del cur_dict_filter[int(neighb)]

                    sorted_dict = sorted(cur_dict.items(), key=lambda item: item[1], reverse=True)
                    sorted_dict_filter = sorted(cur_dict_filter.items(), key=lambda item: item[1], reverse=True)
                    target = cur_logit[test_set[batch_size*i + j][tail]]
                    for rank in range(len(cur_dict)):
                        if sorted_dict[rank][1] - target < 1.0000001e-17:
                            rank += 1
                            rank_list.append(rank)
                            recipr = 1.0/rank
                            rank_recipr_list.append(recipr)
                            break    
                    for rank_filter in range(len(cur_dict_filter)):
                        if sorted_dict_filter[rank_filter][1] - target < 1.0000001e-17:
                            rank_filter += 1
                            rank_list_filter.append(rank_filter)
                            recipr_filter = 1.0/rank_filter
                            rank_recipr_list_filter.append(recipr_filter)
                            break
                if (i+1) % 100 == 0:

                    print("finish batch_no %d" % (i+1))        
                    print("len(rank_recipr_list): ", len(rank_recipr_list))            
                    print("MRR for tail substitution with no type constraint: %f" % (sum(rank_recipr_list)/len(rank_recipr_list)))
    
    
    
        #print("tail substitution with no type constraint:")
        #print(rank_list)
        #print(rank_list_filter)
        #print("MR and MR_filter: %f %f" % (sum(rank_list)/len(rank_list), sum(rank_list_filter)/len(rank_list_filter)))
        #print("MRR and MRR_filter: %f %f" % (sum(rank_recipr_list)/len(rank_recipr_list), sum(rank_recipr_list_filter)/len(rank_recipr_list_filter)))
        count_10 = 0
        count_3 = 0
        count_1 = 0
        count_10_filter = 0
        count_3_filter = 0
        count_1_filter = 0
        print("len(rank_list)", len(rank_list))
        print("len(rank_list_filter)", len(rank_list_filter))
        for i in range(len(rank_list)):
            if rank_list[i] <= 10:
                count_10 += 1
            if rank_list[i] <= 3:
                count_3 += 1
            if rank_list[i] == 1:
                count_1 += 1
        for i in range(len(rank_list_filter)):
            if rank_list_filter[i] <= 10:
                count_10_filter += 1
            if rank_list_filter[i] <= 3:
                count_3_filter += 1
            if rank_list_filter[i] == 1:
                count_1_filter += 1        
        #print("hits@10: %f %f" % (count_10/len(rank_list), count_10_filter/len(rank_list_filter)))
        #print("hits@3: %f %f" % (count_3/len(rank_list), count_3_filter/len(rank_list_filter)))
        #print("hits@1: %f %f" % (count_1/len(rank_list), count_1_filter/len(rank_list_filter)))
        path_OUT = './L_or_R' + str(L_or_R) + "_list_of_rank.txt"
        with open(path_OUT,'w') as f_out:

            for item in rank_list:
                f_out.write("%s\n" % item)
        #metric_dict = {MR:0.0, MRR:0.0, hit10:0.0, hit3:0.0, hit1:0.0,
                       #MR_filter:0.0, MRR_filter:0.0, hit10_filter:0.0, hit3_filter:0.0, hit1_filter:0.0}
        metric_dict = {}
        metric_dict['MR'] = sum(rank_list)/len(rank_list)
        metric_dict['MR_filter'] = sum(rank_list_filter)/len(rank_list_filter)
        metric_dict['MRR'] = sum(rank_recipr_list)/len(rank_recipr_list)
        metric_dict['MRR_filter'] = sum(rank_recipr_list_filter)/len(rank_recipr_list_filter)
        metric_dict['hit10'] = count_10/len(rank_list)
        metric_dict['hit10_filter'] = count_10_filter/len(rank_list_filter)
        metric_dict['hit3'] = count_3/len(rank_list)
        metric_dict['hit3_filter'] = count_3_filter/len(rank_list_filter)
        metric_dict['hit1'] = count_1/len(rank_list)
        metric_dict['hit1_filter'] = count_1_filter/len(rank_list_filter)
        return metric_dict
    
    r_metr = calculate_metrics(0, r_rank_list, r_rank_recipr_list, r_rank_list_filter, r_rank_recipr_list_filter)
    l_metr = calculate_metrics(1, l_rank_list, l_rank_recipr_list, l_rank_list_filter, l_rank_recipr_list_filter)
    print("no type constraint results:")
    print("metric: \t\t MRR \t\t MR \t\t hits@10 \t hits@3 \t hits@1")
    print("l(raw): \t\t %f \t %f \t %f \t %f \t %f" % (l_metr['MRR'], l_metr['MR'], l_metr['hit10'], l_metr['hit3'], l_metr['hit1']))
    print("r(raw): \t\t %f \t %f \t %f \t %f \t %f" % (r_metr['MRR'], r_metr['MR'], r_metr['hit10'], r_metr['hit3'], r_metr['hit1']))
    print("averaged(raw):  \t %f \t %f \t %f \t %f \t %f" % ((l_metr['MRR']+r_metr['MRR'])/2, (l_metr['MR']+r_metr['MR'])/2, (l_metr['hit10']+r_metr['hit10'])/2, (l_metr['hit3']+r_metr['hit3'])/2, (l_metr['hit1']+r_metr['hit1'])/2))
    print("l(filter): \t\t %f \t %f \t %f \t %f \t %f" % (l_metr['MRR_filter'], l_metr['MR_filter'], l_metr['hit10_filter'], l_metr['hit3_filter'], l_metr['hit1_filter']))
    print("r(filter): \t\t %f \t %f \t %f \t %f \t %f" % (r_metr['MRR_filter'], r_metr['MR_filter'], r_metr['hit10_filter'], r_metr['hit3_filter'], r_metr['hit1_filter']))
    print("averaged(filter): \t %f \t %f \t %f \t %f \t %f" % ((l_metr['MRR_filter']+r_metr['MRR_filter'])/2, (l_metr['MR_filter']+r_metr['MR_filter'])/2, (l_metr['hit10_filter']+r_metr['hit10_filter'])/2, (l_metr['hit3_filter']+r_metr['hit3_filter'])/2, (l_metr['hit1_filter']+r_metr['hit1_filter'])/2))
    
    '''
    flag = 1
    if flag == 0:
        G.preprocess_transition_probs()
        walks = G.simulate_walks(args.num_walks, args.walk_length)
        file_output = open("./output201807131401token_walks_not_id.txt", 'wb')
        pickle.dump(walks,file_output)
    
    else:
        file_output = open("./output.txt", 'rb')
        walks = pickle.load(file_output)
        int_walks = []
        for walk in walks:
            int_walk = [int(x) for x in walk]
            int_walks.append(int_walk)
        
        print(type(int_walks[0][1]))
        array_walks = np.asarray(int_walks)
        print(array_walks[0][1])
        print(array_walks[0][2])
        print(array_walks[0][3])
        print(array_walks.shape)
        file_output_20180713 = open("./random_walks_X_ids.txt",'wb')
        pickle.dump(array_walks,file_output_20180713)
        file_output_20180713.close()
        
        
    #walks = [[14165,25,12],[12,25,14141]]

    #sess = tf.Session()
    #saver = tf.train.import_meta_graph('/home/why2011btv/research/TensorFlow-TransX-master/model.vec.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('/home/why2011btv/research/TensorFlow-TransX-master/'))
    #graph = tf.get_default_graph()
    #w1 = graph.get_tensor_by_name("model/ent_embedding:0")
    #print(sess.run(tf.shape(w1)))   #output:[14951   100]
    #w2 = graph.get_tensor_by_name("model/rel_embedding:0")
    #print(sess.run(tf.shape(w2)))   #output:[1345   100]
    
    #pickle.dump(walks, file_output)
    #
    #file_output.write(str(walks))
    file_output.close()
    #learn_embeddings(walks)
    print(str(datetime.now()))
    '''

    
    
# to run the code: python src/main.py --input graph/train2id.edgelist --output emb/karate.emd --relation
if __name__ == "__main__":
    #args = parse_args()
    main()

