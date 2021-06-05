from multiprocessing.connection import Connection
import torch
import pickle


def worker(model: torch.nn.Module, epochs_count: int, master_send_chan: Connection):
    for epoch in range(epochs_count):
        data = pickle.dumps(model)
        master_send_chan.send(data)
