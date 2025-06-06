##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Adapted to tecorigin hardware
import os
import time
import json
import logging
import argparse

import torch
#导入torch_sdaa模块
import torch_sdaa
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from resnest.torch.config import get_cfg
from resnest.torch.models.build import get_model
from resnest.torch.datasets import get_dataset
from resnest.torch.transforms import get_transform
from resnest.torch.loss import get_criterion
from resnest.torch.utils import (save_checkpoint, accuracy,
        AverageMeter, LR_Scheduler, torch_dist_sum, mkdir,
        cached_log_stream, PathManager)

#导入torch.sdaa中的amp库
from torch.sdaa import amp 

import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
train_losses=[]
class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='ResNeSt Training')
        parser.add_argument('--config-file', type=str, default='./config50.yaml',
                            help='training configs')
        # parser.add_argument('--config-file', type=str, default=None,
        #                     help='training configs')
        parser.add_argument('--outdir', type=str, default='/data/ckpt',
                            help='output directory')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        # distributed
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        # parser.add_argument('--dist-backend', default='nccl', type=str,
        #                     help='distributed backend')
        parser.add_argument('--dist_backend', default='tccl', type=str,
                            help='distributed backend')
        # evaluation option
        parser.add_argument('--eval-only', action='store_true', default= False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def main():
    args = Options().parse()
    
    #初始计算cuda数量，torch.cuda.device_count()
    #ngpus_per_node = torch.cuda.device_count()
    
    #修改后计算sdaa数量
    ngpus_per_node = torch.sdaa.device_count()
    
    args.world_size = ngpus_per_node * args.world_size
    # load config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.OPTIMIZER.LR = cfg.OPTIMIZER.LR * args.world_size
    
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

def main_worker(gpu, ngpus_per_node, args, cfg):
    # 修改后设置当前进程的设备为sdaa
    torch.sdaa.set_device(gpu)
    
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    logger.info(f'rank: {args.rank} / {args.world_size}')
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    
    #修改前设置当前进程的设备为cuda
    #torch.cuda.set_device(args.gpu)
    
    #修改后，需要放置到最前面的部分
    # torch.sdaa.set_device(gpu)
    
    if args.gpu == 0:
        mkdir(args.outdir)
        filename = os.path.join(args.outdir, 'train_sdaa_3rd_resnest101.txt')
        fh = logging.StreamHandler(cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
        )
        fh.setFormatter(plain_formatter)
        logger.info(args)
    
    # init the global
    global best_pred, acclist_train, acclist_val

    # seed
    torch.manual_seed(cfg.SEED)
    
    #修改前设置cuda下的随机种子
    #torch.cuda.manual_seed(cfg.SEED)
    #修改后设置sdaa下的随机种子
    torch.sdaa.manual_seed(cfg.SEED)

    # init dataloader
    transform_train, transform_val = get_transform(cfg.DATA.DATASET)(
            cfg.DATA.BASE_SIZE, cfg.DATA.CROP_SIZE, cfg.DATA.RAND_AUG)
    trainset = get_dataset(cfg.DATA.DATASET)(root=cfg.DATA.ROOT,
                                             transform=transform_train,
                                             train=True,
                                             download=True)
    valset = get_dataset(cfg.DATA.DATASET)(root=cfg.DATA.ROOT,
                                           transform=transform_val,
                                           train=False,
                                           download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAINING.WORKERS, pin_memory=True,
        sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.TRAINING.TEST_BATCH_SIZE, shuffle=False,
        num_workers=cfg.TRAINING.WORKERS, pin_memory=True,
        sampler=val_sampler)
    
    # init the model
    model_kwargs = {}
    if cfg.MODEL.FINAL_DROP > 0.0:
        model_kwargs['final_drop'] = cfg.MODEL.FINAL_DROP

    if cfg.TRAINING.LAST_GAMMA:
        model_kwargs['last_gamma'] = True

    model = get_model(cfg.MODEL.NAME)(**model_kwargs)
    
    #修改前，加载模型到cuda
    #model.cuda(args.gpu)
    #修改后，加载模型到sdaa
    model.sdaa(args.gpu)
    
    # 使用 DistributedDataParallel 包装模型
    model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion, train_loader = get_criterion(cfg, train_loader, args.gpu)
    
    #修改前，使用criterion.cuda(args.gpu)
    #criterion.cuda(args.gpu)
    #修改后，使用criterion.sdaa(args.gpu)
    criterion.sdaa(args.gpu)
    
    
    #修改前，使用criterion.cuda(args.gpu)
    #criterion.cuda(args.gpu)
    #修改后，使用criterion.sdaa(args.gpu)
    #criterion.sdaa(args.gpu)
    #criterion.sdaa(gpu)
    #criterion = criterion.to(gpu)
    #model = DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    # model = DistributedDataParallel(model, device_ids=[args.gpu])
    #model = DistributedDataParallel(model, device_ids=[gpu])
    
    # criterion and optimizer
    if cfg.OPTIMIZER.DISABLE_BN_WD:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        if args.gpu == 0:
            logger.info(" Weight decay NOT applied to BN parameters ")
            logger.info(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0 },
                                     {'params': rest_params, 'weight_decay': cfg.OPTIMIZER.WEIGHT_DECAY}],
                                    lr=cfg.OPTIMIZER.LR,
                                    momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cfg.OPTIMIZER.LR,
                                    momentum=cfg.OPTIMIZER.MOMENTUM,
                                    weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0:
                #print("=> loading checkpoint '{}'".format(args.resume))
                logger.info(f"=> loading checkpoint '{args.resume}'")
            with PathManager.open(args.resume, "rb") as f:
                checkpoint = torch.load(f)
            cfg.TRAINING.START_EPOCHS = checkpoint['epoch'] + 1 if cfg.TRAINING.START_EPOCHS == 0 \
                    else cfg.TRAINING.START_EPOCHS
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.gpu == 0:
                
                logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            raise RuntimeError (f"=> no resume checkpoint found at '{args.resume}'")

    scheduler = LR_Scheduler(cfg.OPTIMIZER.LR_SCHEDULER,
                             base_lr=cfg.OPTIMIZER.LR,
                             num_epochs=cfg.TRAINING.EPOCHS,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPTIMIZER.WARMUP_EPOCHS)
    
    def train(epoch):
        
        start_time = time.time()  # 记录epoch开始的时间
        train_sampler.set_epoch(epoch)
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        scaler = torch.sdaa.amp.GradScaler()
        total_batches = len(train_loader)  # 总批次数量

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()  # 记录当前批次的开始时间
            scheduler(optimizer, batch_idx, epoch, best_pred)
            #修改前，将数据加载到cuda
            #data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            #修改后，将数据加载到sdaa
            data, target = data.sdaa(args.gpu), target.sdaa(args.gpu)
            #训练的时候修改输入数据格式
            data = data.to(memory_format=torch.channels_last)
            
            with torch.sdaa.amp.autocast():   # 开启AMP环境
                output = model(data)    
                loss = criterion(output, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()    # loss缩放并反向转播
            scaler.step(optimizer)    # 参数更新
            scaler.update()

            if not cfg.DATA.MIXUP:
                acc1 = accuracy(output, target, topk=(1,))
                top1.update(acc1[0], data.size(0))

            losses.update(loss.item(), data.size(0))

            # 计算当前批次的耗时
            batch_time = time.time() - batch_start_time
            # 计算剩余批次的预估时间
            remaining_batches = total_batches - (batch_idx + 1)
            estimated_time_remaining = remaining_batches * batch_time
            if batch_idx % 100 == 0 and args.gpu == 0:
                train_losses.append(losses.avg)
                print(train_losses)
                print('输出结果路径和模型存储路径一致')
                plt.figure()
                plt.plot([x * 100 for x in range(len(train_losses))], train_losses, label='Train Loss')  # 修改了这一行
                plt.xlabel('Batch_idxs')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Loss Curve')
                plt.savefig('./scripts/train_sdaa_3rd_resnest101.png')
                print('Loss_Curve Saved')
            
                if cfg.DATA.MIXUP:
                    logger.info('Batch: %d| Loss: %.3f | Estimated Time Remaining: %.2f seconds' % 
                            (batch_idx, losses.avg, estimated_time_remaining))
                else:
                    logger.info('Batch: %d| Loss: %.3f | Top1: %.3f | Estimated Time Remaining: %.2f seconds' % 
                            (batch_idx, losses.avg, top1.avg, estimated_time_remaining))
                
            
                

        acclist_train += [top1.avg]
        
        end_time = time.time()  # 记录epoch结束的时间
        epoch_time = end_time - start_time  # 计算epoch所需的时间
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f} seconds') 
        
    

    def validate(epoch):
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        global best_pred, acclist_train, acclist_val
        is_best = False
        for batch_idx, (data, target) in enumerate(val_loader):
            #修改前，将数据加载到cuda
            #data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            #修改后，将数据加载到sdaa
            data, target = data.sdaa(args.gpu), target.sdaa(args.gpu)
            
            with torch.no_grad():
                output = model(data)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))

        # sum all
        sum1, cnt1, sum5, cnt5 = torch_dist_sum(args.gpu, top1.sum, top1.count, top5.sum, top5.count)
        top1_acc = sum(sum1) / sum(cnt1)
        top5_acc = sum(sum5) / sum(cnt5)

        if args.gpu == 0:
            
            logger.info('Validation: Top1: %.3f | Top5: %.3f'%(top1_acc, top5_acc))
            if args.eval_only:
                return top1_acc, top5_acc

            # save checkpoint
            acclist_val += [top1_acc]
            if top1_acc > best_pred:
                best_pred = top1_acc 
                is_best = True
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_pred': best_pred,
                    'acclist_train':acclist_train,
                    'acclist_val':acclist_val,
                },
                directory=args.outdir,
                is_best=False,
                filename=f'resnest101_checkpoint_{epoch}.pth')
        return top1_acc.item(), top5_acc.item()

    if args.export:
        if args.gpu == 0:
            with PathManager.open(args.export + '.pth', "wb") as f:
                torch.save(model.module.state_dict(), f)
        return

    if args.eval_only:
        top1_acc, top5_acc = validate(cfg.TRAINING.START_EPOCHS)
        metrics = {
            "top1": top1_acc,
            "top5": top5_acc,
        }
        if args.gpu == 0:
            with PathManager.open(os.path.join(args.outdir, 'metrics.json'), "w") as f:
                json.dump(metrics, f)
        return

    for epoch in range(cfg.TRAINING.START_EPOCHS, cfg.TRAINING.EPOCHS):
        tic = time.time()
        train(epoch)
        if epoch % 10 == 0:
            top1_acc, top5_acc = validate(epoch)
        elapsed = time.time() - tic
        if args.gpu == 0:
            
            logger.info(f'Epoch: {epoch}, Time cost: {elapsed}')

    # final evaluation
    top1_acc, top5_acc = validate(cfg.TRAINING.START_EPOCHS - 1)
    if args.gpu == 0:
        # save final checkpoint
        save_checkpoint({
                'epoch': cfg.TRAINING.EPOCHS - 1,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'acclist_train':acclist_train,
                'acclist_val':acclist_val,
            },
            directory=args.outdir,
            is_best=False,
            filename='checkpoint_final.pth')

        # save final model weights
        with PathManager.open(os.path.join(args.outdir, 'model_weights.pth'), "wb") as f:
            torch.save(model.module.state_dict(), f)

        metrics = {
            "top1": top1_acc,
            "top5": top5_acc,
        }
        with PathManager.open(os.path.join(args.outdir, 'metrics.json'), "w") as f:
            json.dump(metrics, f)

if __name__ == "__main__":
    main()



