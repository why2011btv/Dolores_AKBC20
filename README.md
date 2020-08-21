Repository for "Dolores: Deep Contextualized Knowledge Graph Embeddings" (AKBC 2020)

Steps 
1) Convert entities and relations to ids, e.g. <br>
convert <br>
/m/09v3jyg /m/0f8l9c /film/film/release_date_s./film/film_regional_release_date/film_release_region<br>
to <br>
0, 1, 0<br>
like what is done in <a href="https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/benchmarks/FB15K237/train2id.txt">OpenKE</a><br>
2) Use node2vec to generate paths:<br>
i) input: edgelist, e.g. see <a href="https://github.com/why2011btv/node2vec_20180802/blob/master/graph/train2id.edgelist">train2id.edgelist</a><br>
ii) edit line 99 in <a href="https://github.com/aditya-grover/node2vec/blob/master/src/main.py">main.py</a> to get random walks (paths).<br>
iii) Hyperparameter includes: p, q, num_walks, walk_length
3) Pretrain ELMo-based model M <br>
edit line 40 in ./bin/train_elmo.py to the file which contains the generated training paths<br>
4) Save the model parameter
5) Combine the model with your downstream task model, fine-tune M's parameters, get the contextual representation, and make final predictions <br>
refer to ./bin/run_test.py
