from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time, os
import numpy as np

import ray
import model


def get_args():
    parser = argparse.ArgumentParser(
        description="Run the federated approach.")
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
    parser.add_argument('--personal_lrn',
                        default=1,
                        type=str2bool,
                        help='whether to set the learning rate of the first node to be smaller')
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
                        default=30000,
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
    parser.add_argument(
        '--store_step',
        default=10,
        type=int,
        help='calculate cen_to_opt, apend it to list every store_step')
    parser.add_argument('--save_step',
                        default=10000,
                        type=int,
                        help='save cen_to_opts to file every save_step')
    parser.add_argument('--round',
                        default=0,
                        type=int,
                        help='whether to save current round to file')
    return parser.parse_args()

@ray.remote
class ParameterServer:
    def __init__(self, init_weight):
        self.master_weight = init_weight.copy()

    def get_master_weight(self):
        return self.master_weight

    def set_master_weight(self, diff):
        self.master_weight += diff


@ray.remote
def get_accu_and_loss(ps, args):
    net = model.SimpleCNN(args)
    mnist = model.download_mnist_retry(seed=1111)
    start_time = time.time()
    value = []
    master_weights = []
    current_time = time.time() - start_time
    while current_time < args.stop_time:
        weights = ray.get(ps.get_master_weight.remote())
        master_weights.append((current_time, weights))
        if current_time > 5:
            t, w = master_weights.pop(0)
            net.set_flat(w)
            xs, xy = mnist.test.next_batch(2000)
            accu, loss = net.compute_accuracy_and_loss(xs, xy)
            print()
            # print(['*']*10)
            print('master_time', t, 'accu:', accu, 'testing loss:', loss)
            # print(['*']*10)
            print()
            value.append((t, accu, loss))
            np.save(
                args.save_dir + 'federated_num_worker%d, k_%d, round_%d, net_lrn_%.6f, FL_lrn_%6f' %
                (args.num_workers, args.k, args.round, args.net_lrn, args.lrns[0]), np.array(value))
        time.sleep(1)
        current_time = time.time() - start_time

@ray.remote
def worker_task(ps, current_worker_index, args):
    # Download MNIST.
    mnist = model.download_mnist_retry(seed=current_worker_index + 1)

    # Initialize the model.
    args.lrn = args.lrns[current_worker_index]
    net = model.SimpleCNN(args)
    
    if current_worker_index == 1:
        xs, ys = mnist.train.next_batch(args.batch_sizes[current_worker_index])
        acc, loss = net.compute_accuracy_and_loss(xs, ys)
        stored_losses = [loss]

    step = 0
    start_time = time.time()
    pre_time = time.time()
    while step < args.steps and time.time() - start_time < args.stop_time:
        time.sleep(max(0, args.time_per_batch[current_worker_index]-(time.time()-pre_time)))
        pre_time = time.time()
        weights = ray.get(ps.get_master_weight.remote())
        # Get the current weights from the parameter server.
        net.set_flat(weights)

        # Compute an update and push it to the parameter server.
        xs, ys = mnist.train.next_batch(args.batch_sizes[current_worker_index])
        loss_value, new_weights = net.minimize(xs, ys)
        diff = new_weights - weights
        if current_worker_index != 0:
            time.sleep(0.1)
        if step % 50 == 0:
            print("step", step, "current_worker_index", current_worker_index, "elapsed time is", time.time() - start_time, "loss is", loss_value)
        ps.set_master_weight.remote(diff)
        step += 1


if __name__ == "__main__":
    ray.init()
    args = get_args()
    args.save_dir = './federated_log/'
    args.time_per_batch = {i:1 for i in range(args.num_workers)}
    args.time_per_batch[0] = 0.125
    args.batch_sizes = {i:64 for i in range(args.num_workers)}
    args.batch_sizes[0] = 1
    print(f'batch sizes are {args.batch_sizes}')
    args.lrns = {i:args.net_lrn/5 for i in range(args.num_workers)}
    if args.personal_lrn:
        args.lrns[0] = args.lrns[1]/8
    print(f'learning rates are {args.lrns}')
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    tmp_args = args
    tmp_args.lrn = args.net_lrn
    net = model.SimpleCNN(tmp_args)
    init_weight = net.get_flat()
    ps = ParameterServer.remote(init_weight=init_weight)


    # Create a parameter server with some random weights.
    workers = [worker_task.remote(ps, index, args) for index in range(args.num_workers)]
    get_accu_and_loss.remote(ps, args)
    ray.wait(workers, num_returns=args.num_workers)

