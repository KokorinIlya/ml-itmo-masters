from typing import Callable, Tuple, List, Optional
import torch
from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from sklearn.cluster import KMeans


def determine_size(grad_size: torch.Size, dim_reducer=5) -> Optional[Tuple[int, int, int]]:
    if len(grad_size) == 1:
        return None
    n = grad_size[0]
    m = 1
    for cur_dim in grad_size[1:]:
        m *= cur_dim
    if n < dim_reducer:
        k = 1
    else:
        k = n // dim_reducer
    return n, m, k


class KMeansQuantizator(GradientProcessor):
    def __init__(self, size_determiner: Callable[[torch.Size], Optional[Tuple[int, int, int]]]):
        self.size_determiner = size_determiner

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.clone() for cur_layer in shard_grads]
        result = []
        for cur_grad in shard_grads:
            size_det_res = self.size_determiner(cur_grad.size())
            if size_det_res is None:
                result.append(cur_grad)
            else:
                n, m, k = size_det_res
                assert n * m == cur_grad.numel() and k <= n
                X = cur_grad.reshape(n, m)
                clust = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=10)
                clust.fit(X)
                centers = torch.from_numpy(clust.cluster_centers_).float()
                X_t = torch.zeros_like(X)
                X_t[list(range(n))] = centers[clust.labels_]
                X_t = X_t.reshape(*cur_grad.size())
                result.append(X_t)
        return result