import argparse
import numpy as np

import ray
import model
import time
import os


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description="Run the synchronous parameter "
                                             "server example.")
parser.add_argument("--num_workers", default=20, type=int, help="The number of workers to use.")
parser.add_argument('--threads_per_worker', default=3, type=int, help='number of threads each worker uses')
parser.add_argument('--sleep_mean', default=1, type=float, help='mean sleep time')
parser.add_argument('--lrn', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--round', default=0, type=int, help='whether to save current round to file')
parser.add_argument('--stop_time', default=5000, type=int, help='training time')


@ray.remote
class ParameterServer(object):
    def __init__(self, args):
        self.net = model.SimpleCNN(args)
        self.begin = False

    def apply_gradients(self, *gradients):
        self.net.apply_gradients(np.mean(gradients, axis=0))
        return self.net.variables.get_flat()

    def get_weights(self):
        return self.net.variables.get_flat()

    def set_begin(self):
        self.begin = True

    def get_begin(self):
        return self.begin


@ray.remote
class Worker(object):
    def __init__(self, worker_index, args):
        self.worker_index = worker_index
        self.batch_size = args.batch_size
        self.mnist = model.download_mnist_retry(seed=worker_index)
        self.net = model.SimpleCNN(args)

    def compute_gradients(self, weights):
        self.net.variables.set_flat(weights)
        xs, ys = self.mnist.train.next_batch(self.batch_size)
        return self.net.compute_gradients(xs, ys)


@ray.remote
def get_accu_and_loss(ps, args):
    net = model.SimpleCNN(args)
    mnist = model.download_mnist_retry(seed=1111)

    value = []
    cents = []
    begin = ray.get(ps.get_begin.remote())
    while not begin:
        time.sleep(0.001)
        begin = ray.get(ps.get_begin.remote())

    start_time = time.time()
    while True:
        the_time = time.time()-start_time
        cent = ray.get(ps.get_weights.remote())
        cents.append((the_time, cent))
        print('number of items in the cents', len(cents))
        time.sleep(1)
        if the_time > 5:
            cent_time, cent = cents.pop(0)
            net.set_flat(cent)
            xs, xy = mnist.test.next_batch(10000)
            accu, loss = net.compute_accuracy_and_loss(xs, xy)
            print()
            print('centralized_time', cent_time, 'accu:', accu, 'loss:', loss)
            print()
            value.append((cent_time, accu, loss))
            np.save(args.save_dir+'centralized_num_worker%d, round %d' % (args.num_workers, args.round), np.array(value))


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    args.save_dir = './centralized_log_%.1f/' % args.sleep_mean
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    # Create a parameter server.
    net = model.SimpleCNN(args)
    ps = ParameterServer.remote(args)

    # Create workers.
    workers = [Worker.remote(worker_index, args)
               for worker_index in range(args.num_workers)]

    # Download MNIST.
    mnist = model.download_mnist_retry()

    i = 0
    current_weights = ps.get_weights.remote()
    get_accu_and_loss.remote(ps, args)
    start_t = time.time()
    while time.time() - start_t < args.stop_time:
        ray.wait([current_weights])
        if i == 1:
            start_t = time.time()
            ps.set_begin.remote()
        sleep_time = np.amax(np.random.exponential(args.sleep_mean, args.num_workers))
        time.sleep(sleep_time)
        gradients = [worker.compute_gradients.remote(current_weights)
                     for worker in workers]
        current_weights = ps.apply_gradients.remote(*gradients)
        i += 1
        if i%100 == 0:
            print('elasped time', time.time()-start_t, 'steps is', i)

