#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import collections
import itertools
import os
import math
import torch
import torch_sdaa
import time
import ctypes

from copy import deepcopy
from functools import reduce

from fairseq import data, distributed_utils, options, progress_bar, tasks, utils, bleu, tokenizer
from fairseq.fp16_trainer import FP16Trainer
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.data import dictionary

from mlperf_compliance import mlperf_log

def main(args):
    if not torch.sdaa.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.sdaa.set_device(args.device_id)
    from mlperf_compliance.mlperf_log import transformer_print
    transformer_print(key=mlperf_log.RUN_CLEAR_CACHES) #before this tag we should run clearing caches on the host
    # mlperf compliance synchronization
    if args.distributed_world_size > 1:
        assert(torch.distributed.is_initialized())
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
        torch.sdaa.synchronize()
    transformer_print(key=mlperf_log.RUN_START)
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)
    transformer_print(key=mlperf_log.OPT_NAME, value=args.optimizer)
    transformer_print(key=mlperf_log.OPT_LR, value=args.lr)
    transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=eval(args.adam_betas)[0])
    transformer_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=eval(args.adam_betas)[1])
    transformer_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.adam_eps)
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    result = torch.sdaa.cudart().cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    result = torch.sdaa.cudart().cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    torch.manual_seed(args.seed)
    transformer_print(key=mlperf_log.RUN_SET_RANDOM_SEED, value=args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)
    transformer_print(key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
            value={'alpha':args.lenpen,
                'beam_size':args.beam,
                'extra_decode_length':args.max_len_b,
                'vocab_size':task.target_dictionary.__len__()
                }
            )

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Build trainer
    if args.fp16:
        trainer = FP16Trainer(args, task, model, criterion)
    else:
        if torch.sdaa.get_device_capability(0)[0] >= 7:
            print('| NOTICE: your device may support faster training with --fp16')
        trainer = Trainer(args, task, model, criterion)
    if (args.online_eval or args.target_bleu) and not args.remove_bpe:
        args.remove_bpe='@@ '
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))
    transformer_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.max_tokens)
    transformer_print(key=mlperf_log.INPUT_ORDER)
    # Initialize dataloader
    max_positions = trainer.get_model().max_positions()

    # Send a dummy batch to warm the caching allocator
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small or model reaches target score
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    tgt_bleu = args.target_bleu or math.inf
    current_bleu = 0.0
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    
    ctr = 0
    class DummyEpochBatchIterator:
        def __init__(self, epoch=0):
            self.epoch = epoch
    
    epoch_itr = DummyEpochBatchIterator(0)
    transformer_print(key=mlperf_log.TRAIN_LOOP)
    while lr >= args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update and current_bleu < tgt_bleu:
        transformer_print(key=mlperf_log.TRAIN_EPOCH, value=epoch_itr.epoch)
        import time
        start = time.time()
        epoch_itr = data.EpochBatchIterator(
            dataset=task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            epoch=epoch_itr.epoch if ctr is not 0 else 0
        )
        print("got epoch iterator", time.time() - start)
           
                
        # Load the latest checkpoint if one is available
        if ctr is 0:
            load_checkpoint(args, trainer, epoch_itr)
        
        # train for one epoch
        start = time.time()
        train(args, trainer, task, epoch_itr)
        print("epoch time ", time.time() - start)

        start = time.time()

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # Eval BLEU score
        transformer_print(key=mlperf_log.EVAL_START, value=epoch_itr.epoch)
        if args.online_eval or (not tgt_bleu is math.inf):
            current_bleu = score(args, trainer, task, epoch_itr, args.gen_subset)
            transformer_print(key=mlperf_log.EVAL_ACCURACY, value={'epoch':epoch_itr.epoch, 'value':current_bleu})
            transformer_print(key=mlperf_log.EVAL_TARGET, value=tgt_bleu)
        transformer_print(key=mlperf_log.EVAL_STOP, value=epoch_itr.epoch)

        # Only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # Save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
        
        ctr = ctr + 1
        print("validation and scoring ", time.time() - start)
        
    train_meter.stop()
    transformer_print(key=mlperf_log.RUN_STOP)
    transformer_print(key=mlperf_log.RUN_FINAL)
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()
    progress = progress_bar.build_progress_bar(args, itr, epoch_itr.epoch, no_progress_bar='simple')

    # update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    if args.enable_parallel_backward_allred_opt and update_freq > 1:
        raise RuntimeError('--enable-parallel-backward-allred-opt is incompatible with --update-freq > 1')

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    #begin = time.time()
    #inside = 0
    for i, sample in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        #newbegin = time.time()
        #print("iter time", newbegin - begin, inside, (newbegin - begin - inside)*1000)
        #begin = newbegin
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            # buffer updates according to --update-freq
            trainer.train_step(sample, update_params=False, last_step=(i == len(itr)-1))
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True, last_step=(i == len(itr)-1))

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        if args.profile is not None and i == args.profile:
            import sys
            sys.exit()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break
        #end = time.time()
        #inside = end - begin

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = data.EpochBatchIterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=trainer.get_model().max_positions(),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses

def score(args, trainer, task, epoch_itr, subset):

    begin = time.time()

    if not subset in task.datasets.keys():
        task.load_dataset(subset)

    src_dict = deepcopy(task.source_dictionary) # This is necessary, generation of translations
    tgt_dict = deepcopy(task.target_dictionary) # alters target dictionary messing up with the rest of training

    model = trainer.get_model()

    #mlperf_log.transformer_print(key=mlperf_log.EVAL_SIZE, value=task.dataset(subset).__len__())
    # Initialize data iterator
    itr = data.EpochBatchIterator(
        dataset=task.dataset(subset),
        max_tokens=None,
        max_sentences=max(8,min(math.ceil(1024/args.distributed_world_size),128)),
        max_positions=model.max_positions(),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    translator = SequenceGenerator(
	[model], tgt_dict, beam_size=args.beam,
	stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
	len_penalty=args.lenpen, unk_penalty=args.unkpen,
	sampling=args.sampling, sampling_topk=args.sampling_topk, minlen=args.min_len,
        )
    # Generate and compute BLEU
    dict = dictionary.Dictionary()
    scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
    num_sentences = 0
    has_target = True
    if args.log_translations:
        log = open(os.path.join(args.save_dir, 'translations_epoch{}_{}'.format(epoch_itr.epoch, args.distributed_rank)), 'w+')
    with progress_bar.build_progress_bar(args, itr) as progress:
        translations = translator.generate_batched_itr(
                progress, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=True, timer=gen_timer, prefix_size=args.prefix_size,
                )

        wps_meter = TimeMeter()
        for sample_id, src_tokens, target_tokens, hypos in translations:
            # Process input and grount truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            src_str = src_dict.string(src_tokens, args.remove_bpe)
            if has_target:
                target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

            if args.log_translations:
                log.write('S-{}\t{}\n'.format(sample_id, src_str))
                if has_target:
                    log.write('T-{}\t{}\n'.format(sample_id, target_str))

            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict = None,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe
                        )
                if args.log_translations:
                    log.write('H-{}\t{}\t{}\n'.format(sample_id, hypo['score'], hypo_str))
                    # log.write(str(hypo_tokens))
                    log.write('P-{}\t{}\n'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))

                # Score only the top hypothesis
                if has_target and i==0:
                    sys_tok = tokenizer.Tokenizer.tokenize((hypo_str.lower() if args.ignore_case else hypo_str), dict)
                    ref_tok = tokenizer.Tokenizer.tokenize((target_str.lower() if args.ignore_case else target_str), dict)
                    scorer.add(ref_tok, sys_tok)

            wps_meter.update(src_tokens.size(0))
            progress.log({'wps':round(wps_meter.avg)})
            num_sentences += 1

    if args.distributed_world_size > 1:
        _all_gather_bleu_scorer(scorer)
    if args.log_translations:
        log.close()
    if gen_timer.sum != 0:
        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1./gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(subset, args.beam, scorer.result_string()))

    print('| Eval completed in: {:.2f}s'.format(time.time()-begin))

    return scorer.score(order=4)

def _all_gather_bleu_scorer(scorer):
    stats = distributed_utils.all_gather_list(scorer.stat)
    bleu_stat = bleu.BleuStat()
    bleu_stat.reflen  = reduce(lambda x,y: x+y, [s.reflen for s in stats])
    bleu_stat.predlen = reduce(lambda x,y: x+y, [s.predlen for s in stats])
    bleu_stat.match1  = reduce(lambda x,y: x+y, [s.match1 for s in stats])
    bleu_stat.count1  = reduce(lambda x,y: x+y, [s.count1 for s in stats])
    bleu_stat.match2  = reduce(lambda x,y: x+y, [s.match2 for s in stats])
    bleu_stat.count2  = reduce(lambda x,y: x+y, [s.count2 for s in stats])
    bleu_stat.match3  = reduce(lambda x,y: x+y, [s.match3 for s in stats])
    bleu_stat.count3  = reduce(lambda x,y: x+y, [s.count3 for s in stats])
    bleu_stat.match4  = reduce(lambda x,y: x+y, [s.match4 for s in stats])
    bleu_stat.count4  = reduce(lambda x,y: x+y, [s.count4 for s in stats])
    scorer.stat = bleu_stat

def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'best': save_checkpoint.best,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path)
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
