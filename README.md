Repository for "Dolores: Deep Contextualized Knowledge Graph Embeddings" (AKBC 2020)

Pointers to code: <br>
To use node2vec for generating paths, see main.py in the node2vec repository:<br>https://github.com/why2011btv/node2vec_20180802/blob/master/src/main.py <br>
To train the model: see ./bin/train_elmo.py <br>
To evaluate the model: see ./bin/run_test.py <br>

Steps 
1) Convert entities and relations to ids, e.g. convert <br>
/m/09v3jyg /m/0f8l9c /film/film/release_date_s./film/film_regional_release_date/film_release_region<br>
to 0, 1, 0<br>
like what is done in OpenKE (https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/benchmarks/FB15K237/train2id.txt)<br>
2) Use node2vec to generate paths:<br>
input: edgelist, e.g. see https://github.com/why2011btv/node2vec_20180802/blob/master/graph/train2id.edgelist
edit line 99 in <a href="https://github.com/aditya-grover/node2vec/blob/master/src/main.py">main.py</a>
