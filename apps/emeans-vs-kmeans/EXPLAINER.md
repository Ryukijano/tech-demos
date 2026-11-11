# e-means vs k-means — short explainer

Lloyd k-means picks \(k\) centroids and repeats two steps: assign every point to the nearest centroid, then move each centroid to the mean of its points. One such pass is an epoch. The objective is the mean squared distance from points to their centroids. FAISS has shipped that algorithm for years as `faiss.Kmeans`.

e-means, from Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, and Jégou ([arXiv:2611.09501](https://arxiv.org/abs/2611.09501)), keeps the same objective and changes the update. It walks the data in mini-batches. For each cluster it keeps a decayed count and a decayed sum, and the centroid is their ratio. The decay starts loose and tightens on a cosine schedule, so early batches can move a centroid and later batches settle it. Every five batches, a cluster whose decayed count has fallen below a tenth of the mean is revived by splitting a donor cluster. After the last epoch the centroids are replaced by the plain mean of the points assigned during that epoch. Batch size and the peak step come from a power-law fit, so a caller sets \(k\) and the number of epochs and does not sweep a learning rate.

The authors report that this reaches the loss Lloyd has after 25 epochs in about 1.4 to 2.6 epochs on embedding sets, which is about ten times fewer passes. On one 64-core CPU they also report about 8–15× less wall-clock than FAISS Lloyd. The code is in FAISS 1.16.0 as `faiss.contrib.e_means`. That release is real: the changelog lists it, and `faiss-cpu==1.16.0` imports `NumpyEmeans`.

Two library details differ from the paper's writeup and are left as shipped. The initial decayed count is \(2^{-16}\), not \(10^{-10}\). The shipped batch and step coefficients are the ones in the FAISS readme, and the source notes that they are not identical to Table 2. At dimension 16 or below the library picks a smaller "quality" batch, and at dimension 2 or below it turns revival off.

This demo asks a narrower question on one 4-vCPU machine. FAISS Lloyd runs 25 iterations on the full matrix. e-means runs 2.5 epochs with `batch_law="auto"` and no hand-set batch or step. Both start from the same random training rows. The score is a fresh nearest-centroid mean squared error. The pre-registered bar is: that error within 1% of Lloyd, and the training call in at most a tenth of Lloyd's wall-clock, on three datasets and seeds 0, 1, and 2. A pass needs both. Matching the loss without the 10× clock is a partial. Missing the loss is a fail.

The measured status is filled in after `./run_all.sh` and is recorded in `README.md`.

Links: [paper](https://arxiv.org/abs/2611.09501), [project page](https://e-means.github.io/), [FAISS 1.16.0 tree](https://github.com/facebookresearch/faiss/tree/v1.16.0/contrib/e_means), [bookmark](https://x.com/swann_bessa/status/2120200714630594911) (not fetched).
