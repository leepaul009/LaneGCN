# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
# import horovod.torch as hvd


from torch.utils.data.distributed import DistributedSampler

from utils import Logger, load_pretrain
from comm import get_world_size, get_rank, is_main_process, synchronize, all_gather, reduce_dict
from mpi4py import MPI
import logging
from torch.nn.parallel import DistributedDataParallel

comm = MPI.COMM_WORLD
#hvd.init()
#torch.cuda.set_device(hvd.local_rank())

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)
parser.add_argument("--gpu_beg", type=int, default=0)
parser.add_argument("--gpu_num", type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)


        
def main():
    '''
    seed = hvd.rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    '''
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join("log.txt")),
            logging.StreamHandler(),
        ],
    )


    # Import all settings for experiment.
    args = parser.parse_args()

    cuda_ids = [args.gpu_beg + i for i in range(args.gpu_num)]
    logging.info("visible devices: {}".format(cuda_ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_ids))
    
    torch.cuda.set_device(args.local_rank)

    distributed = args.gpu_num > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://",)

    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()

    if distributed:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)   
    # net = net.to(device)
    if distributed:
        net = DistributedDataParallel(
            net, 
            find_unused_parameters=True,
            device_ids=[args.local_rank], 
            output_device=args.local_rank)
    logging.info("is distributed: {}, use cuda:{}".format(distributed, args.local_rank))
    
    '''
    if config["horovod"]:
        opt.opt = hvd.DistributedOptimizer(
            opt.opt, named_parameters=net.named_parameters()
        )
    '''

    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(get_module(net, distributed), ckpt["state_dict"], distributed)
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])
    '''
    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        val_sampler = DistributedSampler(
            dataset, num_replicas=get_world_size(), rank=get_rank(), # num_replicas=hvd.size(), rank=hvd.rank()
        )
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        val(config, val_loader, net, loss, post_process, 999)
        return
    '''

    # Create log and copy all code
    save_dir = config["save_dir"]
    log = os.path.join(save_dir, "log")
    # if is_main_process():
    if is_main_process():
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sys.stdout = Logger(log)

        src_dirs = [root_path]
        dst_dirs = [os.path.join(save_dir, "files")]
        for src_dir, dst_dir in zip(src_dirs, dst_dirs):
            files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for f in files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    train_sampler = DistributedSampler(
        dataset, num_replicas=get_world_size(), rank=get_rank(),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset = Dataset(config["val_split"], config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank())
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader, distributed)

def get_module(model, distributed):
    if distributed:
        return model.module
    else:
        return model

def worker_init_fn(pid):
    # np_seed = hvd.rank() * 1024 + int(pid)
    np_seed = get_rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None, distributed=False):
    train_loader.sampler.set_epoch(int(epoch))
    net.train()

    num_batches = len(train_loader)
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(
        config["display_iters"] / (get_world_size() * config["batch_size"])
    )
    val_iters = int(config["val_iters"] / (get_world_size() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    for i, data in tqdm(enumerate(train_loader), disable=get_rank()):
        epoch += epoch_per_batch
        data = dict(data)

        output = net(data)
        loss_out = loss(output, data)
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if is_main_process() and (
            num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            save_ckpt(get_module(net, distributed), opt, config["save_dir"], epoch)

        if num_iters % display_iters == 0:
            dt = time.time() - start_time
            metrics = sync(metrics)
            if is_main_process():
                post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()

    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)

    dt = time.time() - start_time
    metrics = sync(metrics)
    if is_main_process():
        post_process.display(metrics, dt, epoch)
    net.train()


def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


def sync(data):
    data_list = comm.allgather(data)
    data = dict()
    for key in data_list[0]:
        if isinstance(data_list[0][key], list):
            data[key] = []
        else:
            data[key] = 0
        for i in range(len(data_list)):
            data[key] += data_list[i][key]
    return data


if __name__ == "__main__":
    main()
