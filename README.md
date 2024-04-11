# Exploring Learned Index Structures

This project explores the use of learned index structures as an alternative to traditional database indexing techniques. Learned indexes utilize machine learning models to efficiently store and access data, potentially outperforming traditional index structures in terms of runtime complexity and storage requirements.

## Introduction

Traditional index structures, such as B-trees, hash maps, and bloom filters, are general-purpose data structures that assume nothing about the data being stored. This prevents them from taking advantage of common patterns found in real-world data. In contrast, learned indexes have the ability to learn their model based on the data being stored, allowing them to learn patterns and create specialized index structures that can outperform traditional structures.

The main challenge addressed in this project is handling inserts and updates for learned indexes, as prior work has focused on read-only, in-memory database systems. We explore ways to efficiently insert new keys into the learned index structure, specifically focusing on the case of appending to a sorted contiguous array in memory.

## API for Models in Framework

**Methods**

`clear()`
- Remove all of the items from the BTree.

`insert(key, value)` -> 0 or 1
- Add an item. Return 1 if the item was added, or 0 otherwise.

`remove(key)` -> 0 or 1 *Optional*
- Remove an item. Return 1 if removed successfully, or 0 otherwise.

`update(collection)`
- Add the items from the given collection.

`predict(key[, default=None])` -> Value or default
- Return the predicted value for the key.

`get(key, prediction[, default=None])` -> Value for key or default
- Return the actual value or the default if the key is not found. Needs a prediction as a starting place for the search.

`has_key(key)` *Optional*
- Return true if the BTree contains the given key.

`__getitem__(key)`
- Returns `get(key)`.

`__setitem__(key, value)`
- Returns `insert(key, value)`.

`__delitem__(key)` *Optional*
- Returns `remove(key)`.

**Properties**

`results`
- Return a dictionary of all the model-specific results to store.

`mean_error`
- The mean error of the model, used by the hybrid model to replace the model with a BTree.

`min_error`
- The minimum error of the model, used as a testing metric.

`max_error`
- The maximum error of the model, used as a testing metric.

## Related Work

- The concept of learned index structures was introduced in the paper "The Case for Learned Index Structures" by Kraska et al. [1].
- Efforts have been made to formalize the benefits and limitations of learned indexes, specifically with bloom filters [2].
- Incremental learning techniques have been applied to deep convolutional neural networks (DCNNs) to adapt to new data without losing the ability to perform on old data [3].
- The Tree-CNN architecture dynamically adds new leaf structures to the network to handle new classes in an incremental learning setting [4].

## Innovation

Our innovation focuses on efficiently appending new keys to a learned index structure. We explore various methods for live retraining of the index as new keys are added, including:

1. Full retraining from scratch every time an append occurs (baseline).
2. Reusing weights before appending to speed up learning of newly appended data.
3. Incremental learning and appends, with relearning on key accesses.
4. Using a specialized tree structure of experts, where a portion of the tree is trained to specialize on the region of newly inserted data.

## Technical Approaches

We created a standard interface to evaluate a wide variety of index structures. The model API is detailed in Appendix A. Key components of our implementation include:

- Testing data generation: Functions to generate keys following particular distributions (random, normal, exponential, and log-normal).
- Testing framework: Allows for controlled and fair comparisons between different models and learning algorithms.
- Models: Implemented various learned index models, including fully-connected layers (FC), fully-connected layers with residual connections (Res), and a hybrid model [1].

## Experiments and Results

We conducted experiments to evaluate the performance of different models, the impact of hybrid model structure, and the effectiveness of different insertion methods. Key findings include:

- The Bits model, which represents keys as binary input, tended to perform better and more consistently than other models.
- ReLU activation generally outperformed linear activation in terms of mean prediction error.
- More complex hybrid model structures did not significantly improve performance but increased training time.
- Naively training on only new data caused the network to forget previously learned indexes, highlighting the need for more sophisticated insertion methods.

## Discussion and Future Work

Our experiments revealed several insights and areas for future exploration:

- Proper timing measurement is crucial for understanding model performance, particularly the distinction between estimation time and local search time.
- Further optimization of search routines and tuning of models could potentially improve inference time and accuracy.
- Exploring alternative initialization methods, optimizers, and loss functions may help address the issue of models getting stuck in local minima during training.
- Investigating advanced activation functions like Leaky ReLU could potentially improve learning performance.

## References

[1] Kraska, T., Beutel, A., Chi, E. H., Dean, J., & Polyzotis, N. (2018). The case for learned index structures. In Proceedings of the 2018 International Conference on Management of Data (pp. 489-504).

[2] Mitzenmacher, M. (2018). A model for learned bloom filters and optimizing by sandwiching. In Advances in Neural Information Processing Systems (pp. 464-473).

[3] Sarwar, S. S., Haque, A., & Vanaret, C. (2017). Incremental learning in deep convolutional neural networks using partial network sharing. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2274-2278). IEEE.

[4] Roy, D., Panda, P., & Roy, K. (2018). Tree-CNN: A deep convolutional neural network for lifelong learning. arXiv preprint arXiv:1802.05800.
