import argparse
import os
import random
import shutil
import time
import warnings

import torch,torch_sdaa
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from csdarknet53 import CsDarkNet53
#from darknet53 import DarkNet53
#from CSPdarknet import CSPDarkNet
import torch_sdaa.amp  as amp

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:65501', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    # 修改后，使用SDAA作为训练设备：
    device = torch.device('sdaa')

    # 修改前的CUDA接口:
    # torch.cuda.is_available()

    # 改为SDAA接口：
    print(torch.sdaa.is_available())

    # 修改前的CUDA接口：
    # torch.cuda.device_count()

    # 修改后CUDA接口：
    torch.sdaa.device_count()


    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.sdaa.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        torch.sdaa.set_device(args.gpu)    
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        # print("=> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        print("=> creating model csdarknet53")
        #model = DarkNet53(num_classes=1000)
        model = CsDarkNet53(num_classes=1000)
        #model = CSPDarkNet([1, 2, 8, 8, 4], num_classes=1000)
        print(model)

    model.to('sdaa')
    # if not torch.sdaa.is_available():
    #     print('using CPU, this will be slow')
    # elif args.distributed:
    #     # For multiprocessing distributed, DistributedDataParallel constructor
    #     # should always set the single device scope, otherwise,
    #     # DistributedDataParallel will use all available devices.
    #     if args.gpu is not None:
    #         torch.sdaa.set_device(args.gpu)
    #         model.sdaa(args.gpu)
    #         # When using a single GPU per process and per
    #         # DistributedDataParallel, we need to divide the batch size
    #         # ourselves based on the total number of GPUs we have
    #         args.batch_size = int(args.batch_size / ngpus_per_node)
    #         args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     else:
    #         model.sdaa()
    #         # DistributedDataParallel will divide and allocate batch_size to all
    #         # available GPUs if device_ids are not set
    #         model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.sdaa.set_device(args.gpu)
    #     model = model.sdaa(args.gpu)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.sdaa()
    #     else:
    #         model = torch.nn.DataParallel(model).sdaa()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to('sdaa')

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'sdaa:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.Resize((256,256)),
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.03, 0.03)),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(292),
            transforms.CenterCrop(256),
            #transforms.Resize((256,256)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename='checkpoint_labelsmoothing_new.pth.tar')

  
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lr = AverageMeter('Lr', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, lr],
        prefix="Epoch: [{}]".format(epoch))

    
    # switch to train mode
    model.train()

    end = time.time()

    scaler = amp.GradScaler()              # 添加GradScaler
 


    import matplotlib.pyplot as plt

    # 用于保存每一步的损失值
    lossplt = []

    # 创建保存图像的文件夹（如果不存在）
    output_dir = 'output-csp-runtest'
    os.makedirs(output_dir, exist_ok=True)

    # 设置绘图环境
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()  # 创建一个图形对象和一个子图

    # 设置图的标签
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.sdaa(args.gpu, non_blocking=True)
        # if torch.sdaa.is_available():
        #     target = target.sdaa(args.gpu, non_blocking=True)
        images = images.to('sdaa',non_blocking=True)
        target = target.to('sdaa',non_blocking=True)
        images = images.to(memory_format=torch.channels_last) # 数据格式转换为NHWC（channels_last）
        # ------------------------------------
        N = target.size(0)
        C = 1000
        # C is the number of classes.
        smoothed_labels = torch.full(size=(N, C), fill_value=0.1 / (C - 1)).sdaa()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=0.9)
        # ------------------------------------
        with amp.autocast():
            # compute output
            output = model(images)
            
            # ------------------------------------
            log_prob = torch.nn.functional.log_softmax(output, dim=1)
            loss = -torch.sum(log_prob * smoothed_labels) / N
            #loss = criterion(output, target)  # 计算损失
        
        
        
        #loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        lr.update(optimizer.param_groups[0]['lr'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lossplt.append(loss.item())
            
        # 更新loss图
        ax.plot(lossplt, label='Loss', color='blue')
        #ax.legend()
        plt.draw()  # 更新图像
        plt.pause(0.1)  # 暂停一下以更新图表

        # 保存图像到文件
        if i % 10 ==0:
            plt.savefig(os.path.join(output_dir, f'loss_epoch{epoch}_step{i}.png'))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.sdaa(args.gpu, non_blocking=True)
            if torch.sdaa.is_available():
                target = target.sdaa(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_labelsmoothing_new.pth.tar'):
    if (os.environ.get('RANK') == '0'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best_labelsmoothing_new.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
            
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
