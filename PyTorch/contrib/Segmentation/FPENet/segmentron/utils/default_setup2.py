import os
import logging
import json
import torch
import torch_sdaa

from .distributed import get_rank, synchronize
from .logger import setup_logger
from .env import seed_all_rng
from ..config import cfg

"""
启动方式: torchrun --nproc_per_node 4 train.py
"""
def default_setup(args):
    # os.environ['MASTER_ADDR'] = "127.0.0.1"
    # # 设置IP:PORT，框架启动TCP Store为ProcessGroup服务
    os.environ['MASTER_ADDR'] = 'localhost' # 设置IP
    # 从外部获取local_rank参数
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print("local_rank:",local_rank)
    args.local_rank = local_rank
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    # print("dsi11111111111:",args.distributed)
    #print("args.distributed",args.distributed)

    if not args.no_cuda and torch.cuda.is_available():
        # cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        args.device = "cuda"
    elif not args.no_cuda and torch.sdaa.is_available():
        # cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = True
        # args.device = "sdaa"
        # torch.sdaa.set_device(args.local_rank)
        if args.distributed:

            # torch.distributed.init_process_group(backend="tccl", init_method="tcp://127.0.0.1:28765",rank=args.local_rank,world_size=args.world_size)
            # rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
            args.device = torch.device(f"sdaa:{local_rank}")
            torch.sdaa.set_device(args.device)
            # 初始化ProcessGroup，通信后端选择tccl
            torch.distributed.init_process_group(backend="tccl", init_method="env://")
            # world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
            # torch.distributed.init_process_group("tccl", rank=local_rank, world_size=world_size)
            #synchronize()
        else:
            # args.device = "sdaa"
            args.device = torch.device(f"sdaa:{local_rank}")
            torch.sdaa.set_device(args.device)
    else:
        args.distributed = False
        args.device = "cpu"
    # args.distributed = False
    # args.device = "cpu"

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.init_process_group(backend="nccl", init_method="env://")
    #     synchronize()

    # TODO
    # if args.save_pred:
    #     outdir = '../runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
    #     if not os.path.exists(outdir):
    #         os.makedirs(outdir)

    save_dir = cfg.TRAIN.LOG_SAVE_DIR if cfg.PHASE == 'train' else None
    setup_logger("Segmentron", save_dir, get_rank(), filename='train_sdaa_3rd.log')

    logging.info("Using {} GPUs".format(num_gpus))
    # logging.info(args)
    # logging.info(json.dumps(cfg, indent=8))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + get_rank())
