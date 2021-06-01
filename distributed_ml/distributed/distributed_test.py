from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import time
import random


def get_sum(seed):
    cur_res = 0
    random.seed(seed)
    for _ in range(2_000_000):
        cur_res += random.randint(-1_000_000, 1_000_000)
    return cur_res


def get_sum_parallel(seed, pipe: Connection):
    cur_res = get_sum(seed)
    pipe.send((seed, cur_res))


def parallel(seeds):
    recv_conn, send_conn = Pipe(duplex=False)
    proc = [Process(target=get_sum_parallel, args=(seed, send_conn)) for seed in seeds]
    for p in proc:
        p.start()
    res = {}
    for _ in range(len(seeds)):
        seed, cur_res = recv_conn.recv()
        res[seed] = cur_res
    for p in proc:
        p.join()
    recv_conn.close()
    send_conn.close()
    print(res)


def sequential(seeds):
    res = {}
    for seed in seeds:
        res[seed] = get_sum(seed)
    print(res)


def measure_time(action):
    seeds = [24, 42, 56, 74]
    start_time = time.time()
    action(seeds)
    finish_time = time.time()
    print(f"Finished in {finish_time - start_time}")


def main_measure():
    measure_time(sequential)
    measure_time(parallel)


def f(lst, i):
    lst[0] = i
    for _ in range(10_000):
        print(f"{i}, {lst}\n", end='', flush=True)


def main_check_mut():
    lst = [100, 200, 300, 400, 500]
    proc = [Process(target=f, args=(lst, i)) for i in range(2)]
    for p in proc:
        p.start()
    for p in proc:
        p.join()


if __name__ == '__main__':
    main_check_mut()
