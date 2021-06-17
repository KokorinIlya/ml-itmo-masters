import torch


def check_models(base_model: torch.nn.Module, cur_model: torch.nn.Module, eps: float = 1e-4) -> bool:
    if len(list(base_model.parameters())) != len(list(cur_model.parameters())):
        return False
    for base_param, cur_param in zip(base_model.parameters(), cur_model.parameters()):
        cur_dist = (base_param - cur_param).abs().sum()
        if cur_dist > eps:
            return False
    return True
