# Using NMF in text mining

I came up with the idea of using nonnegative matrix factorization as a dimension reduction method for term-document matrices. Obviously, since NMF using random initialization may not be as stable as one would hope, carefully initializing it is what's required. You can read the paper by Boutsidis et al. <a href="#svdinit">[1]</a> on initializing NMF using singular value decomposition.

However, the catch is that numerically, using standard SVD initialization will not result in an appropriate approximation. Therefore I slightly modified the NMF module from Scikit-Learn to use Scipy's sparse SVD instead of the regular SVD.

When NMF decomposes a matrix ![formula](https://render.githubusercontent.com/render/math?math=A) into ![formula](https://render.githubusercontent.com/render/math?math=A=WH) with ![formula](https://render.githubusercontent.com/render/math?math=k) components, taking an argmax of the rows of ![formula](https://render.githubusercontent.com/render/math?math=W) and columns of ![formula](https://render.githubusercontent.com/render/math?math=H) can be used to "cluster" the rows and columns of ![formula](https://render.githubusercontent.com/render/math?math=A) respectively.

`NMFCluster` uses NMF to cluster the samples (documents) by applying the argmax operator to the matrix ![formula](https://render.githubusercontent.com/render/math?math=W), while also clustering the terms into "topics" by applying the same strategy to ![formula](https://render.githubusercontent.com/render/math?math=H).

`TermDocumentReduce` uses `NMFCluster` to group the features (terms) into topics, and then the feature (term) vectors in each topic will be combined into a single vector thus forming a new ![formula](https://render.githubusercontent.com/render/math?math=k) - dimensional space.

Find out more about the theoretical explanation for this use of operators and how it relates to clustering <a href="https://en.wikipedia.org/wiki/Non-negative_matrix_factorization#Clustering_property">here</a>.



## Dependencies
- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Scikit-Learn (>= 0.23)


## Examples
### SparseNMF
This particular class performs exactly the same as the one from Scikit-Learn, and takes the same set of arguments, except for `init`. That is because the initialization is based on SparseNMF by default.
```python3
from SparseNMF import SparseNMF
nmf = SparseNMF(n_components=10)
W = nmf.fit_transform(X)
H = nmf.components_
```

### NMFClustering
You are required to vectorize your documents first (i.e. using CountVectorizer or TFIDFVectorizer from Scikit-Learn).

Assuming `X` is a transposed term-document matrix (![formula](https://render.githubusercontent.com/render/math?math={d \times t})):
#### NMF Clustering Method
```python3
from NMFClustering import NMFCluster
nc = NMFCluster(n_clusters=k)
nc.fit(X)
DocumentClusterAssignments = nc.labels_
TermClusterAssignments = nc.topics_
```
`DocumentClusterAssignments` will be a ![formula](https://render.githubusercontent.com/render/math?math=d) - dimensional vector representing the index of the cluster to which each document is assigned.
`TermClusterAssignments` on the other hand will be a ![formula](https://render.githubusercontent.com/render/math?math=t) - dimensional vector representing the index of the cluster (symbolic topic) to which each term is assigned.
#### Term-Document Feature Reduction
```python3
from NMFClustering import TermDocumentReduce
tr = TermDocumentReduce(n_components=k)
X_10 = tr.fit_transform(X_tfidf)
```
It is highly recommended to normalize your data when clustering using Euclidean distance (use scikit's `Normalizer`).


## References
<div id="svdinit">
[1] Boutsidis, Christos, and Efstratios Gallopoulos. "SVD based initialization: A head start for nonnegative matrix factorization." Pattern recognition 41, no. 4 (2008): 1350-1362.
</div>
