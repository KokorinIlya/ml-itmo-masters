from typing import Callable, Tuple, List, Optional, Iterable
import torch
from distributed_ml.grad_processor.gradient_processor import GradientProcessor
from sklearn.cluster import KMeans
from distributed_ml.grad_processor.utils import get_flattened_grads, get_unflattened_grads


def __get_prod(lst: Iterable[int]) -> int:
    res = 1
    for x in lst:
        res *= x
    return res


def determine_size_per_layer(grad_size: torch.Size) -> Optional[Tuple[int, int, int, int]]:
    if len(grad_size) != 4:
        return None
    n = __get_prod(grad_size[:2])
    m = __get_prod(grad_size[2:])
    dim_reducer = 100
    if n < dim_reducer:
        k = 1
    else:
        k = n // dim_reducer
    return n, m, k, 0


def determine_size_total_cut_reminder(grad_size: torch.Size) -> Tuple[int, int, int, int]:
    assert grad_size == torch.Size([175722])
    n = 17570
    m = 10
    r = 22
    k = n // 500
    return n, m, k, r


def determine_size_total_no_cut(grad_size: torch.Size) -> Tuple[int, int, int, int]:
    assert grad_size == torch.Size([175722])
    n = 29287
    m = 2 * 3
    r = 0
    k = n // 1000
    return n, m, k, r


class KMeansQuantizator(GradientProcessor):
    def __init__(self, per_layer: bool,
                 size_determiner: Callable[[torch.Size], Optional[Tuple[int, int, int, int]]]):
        self.size_determiner = size_determiner
        self.per_layer = per_layer

    @staticmethod
    def __process(grad: torch.Tensor, n: int, m: int, k: int, r: int) -> torch.Tensor:
        assert n * m + r == grad.numel() and k <= n
        flat_grad = grad.flatten()
        rem_grad = flat_grad[n * m:]
        X = flat_grad[:n * m].reshape(n, m)
        clust = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=10)
        clust.fit(X)
        centers = torch.from_numpy(clust.cluster_centers_).float()
        X_clust = torch.zeros_like(X)
        # noinspection PyTypeChecker
        X_clust[range(n)] = centers[clust.labels_]
        res_grad = torch.hstack((X_clust.flatten(), rem_grad))
        return res_grad.reshape(*grad.size())

    def __do_per_layer(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        result = []
        for cur_grad in shard_grads:
            size_det_res = self.size_determiner(cur_grad.size())
            if size_det_res is None:
                result.append(cur_grad)
            else:
                n, m, k, r = size_det_res
                res_grad = KMeansQuantizator.__process(cur_grad, n, m, k, r)
                result.append(res_grad)
        return result

    def __do_total(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        flattened_grads = get_flattened_grads(shard_grads)
        size_det_res = self.size_determiner(flattened_grads.size())
        assert size_det_res is not None
        n, m, k, r = size_det_res
        flattened_res = KMeansQuantizator.__process(flattened_grads, n, m, k, r)
        return get_unflattened_grads(shard_grads, flattened_res)

    def __call__(self, shard_grads: List[torch.Tensor]) -> List[torch.Tensor]:
        shard_grads = [cur_layer.detach().clone() for cur_layer in shard_grads]
        if self.per_layer:
            return self.__do_per_layer(shard_grads)
        else:
            return self.__do_total(shard_grads)
