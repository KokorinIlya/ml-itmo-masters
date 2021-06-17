import torch


def check_models(base_model: torch.nn.Module, cur_model: torch.nn.Module, eps: float = 1e-4) -> bool:
    base_params = list(base_model.parameters())
    cur_params = list(cur_model.parameters())
    if len(base_params) != len(cur_params):
        return False
    for base_param, cur_param in zip(base_params, cur_params):
        cur_dist = (base_param - cur_param).abs().sum()
        if cur_dist > eps:
            return False
    return True
