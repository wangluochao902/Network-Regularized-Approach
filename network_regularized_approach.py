import argparse
import time
import numpy as np
import os
import ray
import model
import networkx as nx


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the network regularized approach.")
    parser.add_argument('--num_workers',
                        default=5,
                        type=int,
                        help='number of processes to train with')
    parser.add_argument('--threads_per_worker',
                        default=1,
                        type=int,
                        help='number of threads each worker uses')
    parser.add_argument('--tf_seed', default=1, type=int, help='seed')
    parser.add_argument('--net_lrn',
                        default=0.01,
                        type=float,
                        help='learning rate of network regularized approach')
    parser.add_argument(
        '--personal_lrn',
        default=0,
        type=str2bool,
        help='whether to set the learning rate of the first node to be smaller')
    parser.add_argument('--a', default=10, type=float, help='attraction')
    parser.add_argument('--graph_seed',
                        default=1,
                        type=int,
                        help='random seed to generate the graph')
    parser.add_argument('--k',
                        default=4,
                        type=int,
                        help='each worker is connected with k nearest '
                        'neighbors in watts_strogatz_graph')
    parser.add_argument('--print_steps',
                        default=100,
                        type=int,
                        help='print out result to screen every print_steps')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--steps',
                        default=15000,
                        type=int,
                        help='number of steps')
    parser.add_argument('--num_epochs',
                        default=20,
                        type=int,
                        help='number of epochs')
    parser.add_argument('--stop_time',
                        default=1500,
                        type=int,
                        help='training time')
    parser.add_argument('--round',
                        default=0,
                        type=int,
                        help='whether to save current round to file')
    return parser.parse_args()


def construct_graph_watts(num_worker, k, seed):
    g = nx.watts_strogatz_graph(num_worker, k, 0, seed=seed)
    d = dict()
    for node in range(num_worker):
        d[node] = list(g.neighbors(node))
    return d


def get_sleep_time(t):
    if t < 10:
        return 1
    if t < 100:
        return 3
    if t < 300:
        return 6
    if t < 1000:
        return 20
    return 25


@ray.remote
class ParameterServer:

    def __init__(self, num_workers, weights_ids, graph):
        self.loss = [None for _ in range(num_workers)]
        self.weights_ids = weights_ids
        self.graph = graph

    def get_weights_ids(self):
        return self.weights_ids

    def set_weights_ids(self, worker_index, id):
        self.weights_ids[worker_index] = id[0]

    def get_loss(self):
        return self.loss

    def set_loss(self, worker_index, value):
        self.loss[worker_index] = value

    def get_graph(self):
        return self.graph


@ray.remote
def get_accu_and_loss(ps, args):
    net = model.SimpleCNN(args)
    mnist = model.download_mnist_retry(seed=1111)

    # before we start the training, check all the loss value is set which means all workers are ready
    while True:
        losses = ray.get(ps.get_loss.remote())
        if None not in losses:
            print("begin")
            start_time = time.time()
            break
        else:
            time.sleep(0.0001)

    value = []
    cents = []
    current_time = time.time() - start_time
    while current_time < args.stop_time:
        all_weights_ids = ray.get(ps.get_weights_ids.remote())
        all_weights = np.array(
            [ray.get(all_weights_ids[i]) for i in range(args.num_workers)])
        cent = np.mean(all_weights, axis=0)
        cents.append((current_time, cent))
        if current_time > 5:
            cent_time, cent = cents.pop(0)
            net.set_flat(cent)
            xs, xy = mnist.test.next_batch(10000)
            accu, loss = net.compute_accuracy_and_loss(xs, xy)
            print()
            # print(['*']*10)
            print('cent_time', cent_time, 'accu:', accu, 'testing loss:', loss)
            # print(['*']*10)
            print()
            value.append((cent_time, accu, loss))
            np.save(
                args.save_dir +
                'flocking_num_worker%d, k_%d, round_%d, net_lrn_%.6f, node0_lrn_%6f, attraction_%.4f_center_v1'
                % (args.num_workers, args.k, args.round, args.net_lrn,
                   args.lrns[0], args.a), np.array(value))
        time.sleep(1)
        current_time = time.time() - start_time


@ray.remote
def worker_task(ps, current_worker_index, args):
    mnist = model.download_mnist_retry(seed=current_worker_index + 1)

    # Initialize the model.
    args.lrn = args.lrns[current_worker_index]
    net = model.SimpleCNN(args)
    xs, ys = mnist.train.next_batch(args.batch_size[current_worker_index])
    loss_value, _ = net.minimize(xs, ys)

    all_weights_ids = ray.get(ps.get_weights_ids.remote())
    new_weights = ray.get(all_weights_ids[current_worker_index])
    net.set_flat(new_weights)
    ps.set_loss.remote(current_worker_index, loss_value)

    # before we start the training, check all the loss value is set which means all workers are ready
    while True:
        losses = ray.get(ps.get_loss.remote())
        if None not in losses:
            print("begin")
            start_time = time.time()
            break
        else:
            time.sleep(0.0001)

    flocking_group = ray.get(ps.get_graph.remote())[current_worker_index]
    step = 0

    def get_flocking_potential(weights):
        all_weights_ids = ray.get(ps.get_weights_ids.remote())
        flocking_dis = []
        for fw in flocking_group:
            w = ray.get(all_weights_ids[fw])
            # check whether there is nan in the weights. For debugging purpose
            # if np.isnan(np.min(w)):
            #     print('\n\n\n\n\n\n\n\n\n\nthere is nan in weights')
            #     print(ray.get(all_weights_ids[fw]))
            #     print('fw is', fw)
            #     print(weights)
            #     print('current_worker_index is', current_worker_index)
            #     return
            flocking_dis.append(weights - w)
        return np.sum(np.array(flocking_dis), axis=0) * args.a

    start_time = time.time()
    pre_time = time.time()
    next_weigth_save_time = start_time
    while step < args.steps and time.time() - start_time < args.stop_time:
        time.sleep(
            max(
                0, args.time_per_batch[current_worker_index] -
                (time.time() - pre_time)))
        pre_time = time.time()
        xs, ys = mnist.train.next_batch(args.batch_size[current_worker_index])

        loss_value, new_weights = net.minimize(xs, ys)
        ps.set_loss.remote(current_worker_index, loss_value)
        weights = new_weights
        f_p = get_flocking_potential(weights)
        new_weights = net.get_flat()
        new_weights -= args.lrn * f_p
        net.set_flat(new_weights)
        weights_id = ray.put(new_weights)
        ps.set_weights_ids.remote(current_worker_index, [weights_id])
        step += 1
        # if step % 100 == 0 and current_worker_index == 0:
        if step % 100 == 1:
            print('step', step, 'current_worker_index', current_worker_index,
                  'elapsed_time',
                  time.time() - start_time, 'training loss is', loss_value)
        save = True
        if save:
            os.makedirs(args.save_dir + "saved_weight/", exist_ok=True)
        if time.time() > next_weigth_save_time:
            saved_weight = [time.time() - start_time, new_weights]
            np.save(
                args.save_dir +
                'saved_weight/flocking_num_worker%d, k_%d, round_%d, net_lrn_%.6f, node0_lrn_%6f, attraction_%.4f_worker_%d_time_%.2f'
                % (args.num_workers, args.k, args.round,
                   args.net_lrn, args.lrns[0], args.a, current_worker_index,
                   time.time() - start_time), np.array(saved_weight))
            next_weigth_save_time = time.time() + get_sleep_time(time.time() -
                                                                 start_time)


if __name__ == "__main__":
    args = get_args()
    ray.init(num_cpus=args.num_workers + 1)
    args.save_dir = './network_regularized_log/'
    args.time_per_batch = {i: 1 for i in range(args.num_workers)}
    args.time_per_batch[0] = 0.125
    args.lrns = {i: args.net_lrn for i in range(args.num_workers)}
    # if args.personal_lrn:
    #     args.lrns[0] = args.net_lrn/8
    print(f'learning rates are {args.lrns}')

    args.batch_size = {i: 64 for i in range(args.num_workers)}
    args.batch_size[0] = 1
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None

    tmp_args = args
    tmp_args.lrn = args.net_lrn
    net = model.SimpleCNN(tmp_args)
    init_weight = net.get_flat()
    print('\n\n shape is', init_weight.shape, '\n\n')
    weights = [init_weight for _ in range(args.num_workers)]
    weights_ids = [ray.put(w) for w in weights]
    graph = construct_graph_watts(args.num_workers,
                                  args.k,
                                  seed=args.graph_seed)
    print(graph)
    ps = ParameterServer.remote(num_workers=args.num_workers,
                                weights_ids=weights_ids,
                                graph=graph)
    worker_tasks = [
        worker_task.remote(ps, i, args) for i in range(args.num_workers)
    ]
    get_accu_and_loss.remote(ps, args)
    ray.wait(worker_tasks, num_returns=args.num_workers)
