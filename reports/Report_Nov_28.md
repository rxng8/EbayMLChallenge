# Report Nov 28, 2020.

## Birch Algorithm:

### Motivation:
* Since we have to perfectly cluster the dataset, partitioning them into small different clusters based on each category in the text, some kind of tree clustering (heirachical clustering) would be helpful.

### Algorithm:

* I use WordToVec word vectorizer in order to vectorize the word. In order for the vector to be accurate. The word vector model's vocabulary is built only with the the values of that key attribute in every listing.

* My purposed algorithm is to build a Clustering Feature Tree according to the BIRCH algorithm. Here is the birch algorithm: 

* For :
    * D is the dataset containing {t_1, t_2, ..., t_n}

```
for each t_i in D:
    deternmine the correct leaf node for ti insertion
    if threshold condition is not violated:
        add t_i to the cluster and update clustering feature triplets.
    else:
        if there are still room for new cf cluster in the node:
            insert t_i as a single cluster and update clustering feature triplets
        else:
            split the leaf node and redistribute the clustering feature.
```

* In my modified birch algorithms, each depth of the tree represents the partition clustering of a particular listing attributes. Here is the proposed algorithm:
```
add all data in D to the root node of the tree
# looping through each depth
for each (key_attribute of the listings, depth of tree) in zip(total_key_features, birch_tree):
    for each node in that depth:
        for each t_i in D:
            deternmine the correct leaf node for t_i insertion (by calculate the cosine similarity to the centroid (the centroid is the mean vector of every data_point in the cluster, which also is the linear sum of the clustring feature) and compare it to the threshold, in this case 0.5)
            if threshold condition is not violated:
                add t_i to the cluster and update clustering feature triplets.
            else:
                insert t_i as a single cluster and update clustering feature triplets
```



* Another purpose for this algorithm is that we perform multiple Birch partitioning for each of the Birch partition, we cluster according to just one key attribute of the listing. Here is the proposed algorithm:
```

```

----------------
## Research:

* According to my research, two word are considered similarity if the measured cosine similarity of the words is above 0.5. 