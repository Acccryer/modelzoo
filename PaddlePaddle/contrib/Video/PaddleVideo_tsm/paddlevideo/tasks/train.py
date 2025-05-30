# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import time
import copy
import yaml
import errno
from collections import OrderedDict
import sys
import paddle
import paddle_sdaa
import paddle.amp as amp
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from paddle.jit import to_static
from paddlevideo.utils import (
    add_profiler_step,
    build_record,
    get_logger,
    load,
    log_batch,
    log_epoch,
    mkdir,
    save,
)

from ..loader.builder import build_dataloader, build_dataset
from ..metrics.ava_utils import collect_results_cpu
from ..modeling.builder import build_model
from ..solver import build_lr, build_optimizer
from ..utils import do_preciseBN
from .export import get_input_spec
from .download import get_weights_path_from_url
from .save_result import update_train_results, save_model_info


def _mkdir_if_not_exist(path, logger):
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not osp.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and osp.isdir(path):
                logger.warning(
                    "be happy if some process has already created {}".format(path)
                )
            else:
                raise OSError("Failed to mkdir {}".format(path))


def train_model(
    cfg,
    weights=None,
    parallel=True,
    validate=True,
    use_amp=False,
    amp_level=None,
    max_iters=None,
    use_fleet=False,
    profiler_options=None,
):
    """Train model entry

    Args:
        cfg (dict): configuration.
        weights (str, optional): weights path for finetuning. Defaults to None.
        parallel (bool, optional): whether multi-cards training. Defaults to True.
        validate (bool, optional): whether to do evaluation. Defaults to True.
        use_amp (bool, optional): whether to use automatic mixed precision during training. Defaults to False.
        amp_level (str, optional): amp optmization level, must be 'O1' or 'O2' when use_amp is True. Defaults to None.
        max_iters (int, optional): max running iters in an epoch. Defaults to None.
        use_fleet (bool, optional): whether to use fleet. Defaults to False.
        profiler_options (str, optional): configuration for the profiler function. Defaults to None.

    """
    if cfg.get("Global"):
        uniform_output_enabled = cfg.Global.get("uniform_output_enabled", False)
    else:
        uniform_output_enabled = False

    if use_fleet:
        fleet.init(is_collective=True)

    logger = get_logger("paddlevideo")
    batch_size = cfg.DATASET.get("batch_size", 8)
    valid_batch_size = cfg.DATASET.get("valid_batch_size", batch_size)

    # gradient accumulation settings
    use_gradient_accumulation = cfg.get("GRADIENT_ACCUMULATION", None)
    if use_gradient_accumulation and dist.get_world_size() >= 1:
        global_batch_size = cfg.GRADIENT_ACCUMULATION.get("global_batch_size", None)
        num_gpus = dist.get_world_size()

        assert isinstance(
            global_batch_size, int
        ), f"global_batch_size must be int, but got {type(global_batch_size)}"
        assert (
            batch_size <= global_batch_size
        ), f"global_batch_size({global_batch_size}) must not be less than batch_size({batch_size})"

        cur_global_batch_size = (
            batch_size * num_gpus
        )  # The number of batches calculated by all GPUs at one time
        assert (
            global_batch_size % cur_global_batch_size == 0
        ), f"The global batchsize({global_batch_size}) must be divisible by cur_global_batch_size({cur_global_batch_size})"
        cfg.GRADIENT_ACCUMULATION["num_iters"] = (
            global_batch_size // cur_global_batch_size
        )
        # The number of iterations required to reach the global batchsize
        logger.info(
            f"Using gradient accumulation training strategy, "
            f"global_batch_size={global_batch_size}, "
            f"num_gpus={num_gpus}, "
            f"num_accumulative_iters={cfg.GRADIENT_ACCUMULATION.num_iters}"
        )

    if cfg.get("use_npu", False):
        places = paddle.set_device("npu")
    elif cfg.get("use_xpu", False):
        places = paddle.set_device("xpu")
    else:
        places = paddle.set_device("sdaa")

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get("num_workers", 0)
    valid_num_workers = cfg.DATASET.get("valid_num_workers", num_workers)
    model_name = cfg.model_name
    if cfg.get("Global") is not None:
        output_dir = cfg.get("output_dir", f"./output")
    else:
        output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 1. Construct model
    model = build_model(cfg.MODEL)

    if cfg.get("to_static", False):
        specs = None
        model = paddle.jit.to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(specs))

    # 2. Construct dataset and dataloader for training and evaluation
    train_dataset = build_dataset((cfg.DATASET.train, cfg.PIPELINE.train))
    train_dataloader_setting = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn_cfg=cfg.get("MIX", None),
        places=places,
    )
    train_loader = build_dataloader(train_dataset, **train_dataloader_setting)

    if validate:
        valid_dataset = build_dataset((cfg.DATASET.valid, cfg.PIPELINE.valid))
        validate_dataloader_setting = dict(
            batch_size=valid_batch_size,
            num_workers=valid_num_workers,
            places=places,
            drop_last=False,
            shuffle=cfg.DATASET.get(
                "shuffle_valid", False
            ),  # NOTE: attention_LSTM needs to shuffle valid data.
        )
        valid_loader = build_dataloader(valid_dataset, **validate_dataloader_setting)

    # 3. Construct learning rate scheduler(lr) and optimizer
    lr = build_lr(cfg.OPTIMIZER.learning_rate, len(train_loader))
    optimizer = build_optimizer(
        cfg.OPTIMIZER, lr, model=model, use_amp=use_amp, amp_level=amp_level
    )

    # 4. Construct scalar and convert parameters for amp(optional)
    if use_amp:
        scaler = amp.GradScaler(
            init_loss_scaling=2.0**16,
            incr_every_n_steps=2000,
            decr_every_n_nan_or_inf=1,
        )
        # convert model parameters to fp16 when amp_level is O2(pure fp16)
        model, optimizer = amp.decorate(
            models=model,
            optimizers=optimizer,
            level=amp_level,
            master_weight=True,
            save_dtype=None,
        )
        # NOTE: save_dtype is set to float32 now.
        logger.info(f"Training in amp mode, amp_level={amp_level}.")
    else:
        assert (
            amp_level is None
        ), f"amp_level must be None when training in fp32 mode, but got {amp_level}."
        logger.info("Training in fp32 mode.")

    # 5. Resume(optional)
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        filename = osp.join(output_dir, model_name + f"_epoch_{resume_epoch:05d}")
        resume_model_dict = load(filename + ".pdparams")
        resume_opt_dict = load(filename + ".pdopt")
        model.set_state_dict(resume_model_dict)
        optimizer.set_state_dict(resume_opt_dict)
        logger.info("Resume from checkpoint: {}".format(filename))

    # 6. Finetune(optional)
    if weights:
        assert (
            resume_epoch == 0
        ), f"Conflict occurs when finetuning, please switch resume function off by setting resume_epoch to 0 or not indicating it."
        model_dict = load(weights)
        model.set_state_dict(model_dict)
        logger.info("Finetune from checkpoint: {}".format(weights))

    if cfg.get("Global") is not None:
        weight = cfg.Global.get("pretrained_model")

        if weight is not None:
            if weight.startswith(("http://", "https://")):
                weight = get_weights_path_from_url(weight)
            state_dicts = load(weight)
            logger.info(f"Load pretrained model from {weight}")
            model.set_state_dict(state_dicts)

    # 7. Parallelize(optional)
    if parallel:
        model = paddle.DataParallel(model)

    if use_fleet:
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)

    # 8. Train Model
    best = 0.0
    start_time = time.time()
    for epoch in range(0, cfg.epochs):
        time0 = time.time()
        if time0-start_time > 3600*4:
            sys.exit(1)
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch + 1}] <= resume_epoch: [{resume_epoch}], continue..."
            )
            continue
        model.train()

        record_list = build_record(cfg.MODEL)
        tic = time.time()
        for i, data in enumerate(train_loader):
            """Next two line of code only used in test_tipc,
            ignore it most of the time"""
            if max_iters is not None and i >= max_iters:
                break

            record_list["reader_time"].update(time.time() - tic)

            # Collect performance information when profiler_options is activate
            add_profiler_step(profiler_options)

            # 8.1 forward
            # AMP #
            if use_amp:
                with amp.auto_cast(
                    custom_black_list={"reduce_mean", "conv3d"}, level=amp_level
                ):
                    outputs = model(data, mode="train")
                avg_loss = outputs["loss"]
                if use_gradient_accumulation:
                    # clear grad at when epoch begins
                    if i == 0:
                        optimizer.clear_grad()
                    # Loss normalization
                    avg_loss /= cfg.GRADIENT_ACCUMULATION.num_iters
                    # Loss scaling
                    scaled = scaler.scale(avg_loss)
                    # 8.2 backward
                    scaled.backward()
                    # 8.3 minimize
                    if (i + 1) % cfg.GRADIENT_ACCUMULATION.num_iters == 0:
                        scaler.minimize(optimizer, scaled)
                        optimizer.clear_grad()
                else:  # general case
                    # Loss scaling
                    scaled = scaler.scale(avg_loss)
                    # 8.2 backward
                    scaled.backward()
                    # 8.3 minimize
                    scaler.minimize(optimizer, scaled)
                    optimizer.clear_grad()
            else:
                outputs = model(data, mode="train")
                avg_loss = outputs["loss"]
                if use_gradient_accumulation:
                    # clear grad at when epoch begins
                    if i == 0:
                        optimizer.clear_grad()
                    # Loss normalization
                    avg_loss /= cfg.GRADIENT_ACCUMULATION.num_iters
                    # 8.2 backward
                    avg_loss.backward()
                    # 8.3 minimize
                    if (i + 1) % cfg.GRADIENT_ACCUMULATION.num_iters == 0:
                        optimizer.step()
                        optimizer.clear_grad()
                else:  # general case
                    # 8.2 backward
                    avg_loss.backward()
                    # 8.3 minimize
                    optimizer.step()
                    optimizer.clear_grad()

            # log record
            record_list["lr"].update(optimizer.get_lr(), batch_size)
            for name, value in outputs.items():
                if name in record_list:
                    record_list[name].update(value, batch_size)

            record_list["batch_time"].update(time.time() - tic)
            tic = time.time()

            if i % cfg.get("log_interval", 10) == 0:
                ips = "ips: {:.5f} instance/sec,".format(
                    batch_size / record_list["batch_time"].val
                )
                cur_progress = ((i + 1) + epoch * len(train_loader)) / (
                    len(train_loader) * cfg.epochs
                )
                eta = int(
                    record_list["batch_time"].sum * (1 - cur_progress) / cur_progress
                    + 0.5
                )
                log_batch(record_list, i, epoch + 1, cfg.epochs, "train", ips, eta)

            # learning rate iter step
            if cfg.OPTIMIZER.learning_rate.get("iter_step"):
                lr.step()

        # learning rate epoch step
        if not cfg.OPTIMIZER.learning_rate.get("iter_step"):
            lr.step()

        ips = "avg_ips: {:.5f} instance/sec.".format(
            batch_size * record_list["batch_time"].count / record_list["batch_time"].sum
        )
        log_epoch(record_list, epoch + 1, "train", ips)

        def evaluate(best):
            model.eval()
            results = []
            record_list = build_record(cfg.MODEL)
            record_list.pop("lr")
            tic = time.time()
            if parallel:
                rank = dist.get_rank()
            # single_gpu_test and multi_gpu_test
            for i, data in enumerate(valid_loader):
                """Next two line of code only used in test_tipc,
                ignore it most of the time"""
                if max_iters is not None and i >= max_iters:
                    break

                if use_amp:
                    with amp.auto_cast(
                        custom_black_list={"reduce_mean", "conv3d"}, level=amp_level
                    ):
                        outputs = model(data, mode="valid")
                else:
                    outputs = model(data, mode="valid")

                if cfg.MODEL.framework == "FastRCNN":
                    results.extend(outputs)

                # log_record
                if cfg.MODEL.framework != "FastRCNN":
                    for name, value in outputs.items():
                        if name in record_list:
                            record_list[name].update(value, batch_size)

                record_list["batch_time"].update(time.time() - tic)
                tic = time.time()

                if i % cfg.get("log_interval", 10) == 0:
                    ips = "ips: {:.5f} instance/sec.".format(
                        valid_batch_size / record_list["batch_time"].val
                    )
                    log_batch(record_list, i, epoch + 1, cfg.epochs, "val", ips)

            if cfg.MODEL.framework == "FastRCNN":
                if parallel:
                    results = collect_results_cpu(results, len(valid_dataset))
                if not parallel or (parallel and rank == 0):
                    eval_res = valid_dataset.evaluate(results)
                    for name, value in eval_res.items():
                        record_list[name].update(value, valid_batch_size)

            ips = "avg_ips: {:.5f} instance/sec.".format(
                valid_batch_size
                * record_list["batch_time"].count
                / record_list["batch_time"].sum
            )
            log_epoch(record_list, epoch + 1, "val", ips)

            best_flag = False
            if cfg.MODEL.framework == "FastRCNN" and (
                not parallel or (parallel and rank == 0)
            ):
                if record_list["mAP@0.5IOU"].val > best:
                    best = record_list["mAP@0.5IOU"].val
                    best_flag = True
                return best, best_flag

            if cfg.MODEL.framework == "YOWOLocalizer" and (
                not parallel or (parallel and rank == 0)
            ):
                if record_list["fscore"].avg > best:
                    best = record_list["fscore"].avg
                    best_flag = True
                return best, best_flag, record_list["fscore"].avg

            # forbest2, cfg.MODEL.framework != "FastRCNN":
            for top_flag in ["hit_at_one", "top1", "rmse", "F1@0.50"]:
                if record_list.get(top_flag):
                    if top_flag != "rmse" and record_list[top_flag].avg > best:
                        best = record_list[top_flag].avg
                        best_flag = True
                    elif top_flag == "rmse" and (
                        best == 0.0 or record_list[top_flag].avg < best
                    ):
                        best = record_list[top_flag].avg
                        best_flag = True
            acc = record_list["top1"].avg
            return best, best_flag, acc

        # use precise bn to improve acc
        if cfg.get("PRECISEBN") and (
            epoch % cfg.PRECISEBN.preciseBN_interval == 0 or epoch == cfg.epochs - 1
        ):
            do_preciseBN(
                model,
                train_loader,
                parallel,
                min(cfg.PRECISEBN.num_iters_preciseBN, len(train_loader)),
                use_amp,
                amp_level,
            )

        # 9. Validation
        if validate and (
            epoch % cfg.get("val_interval", 1) == 0 or epoch == cfg.epochs - 1
        ):
            with paddle.no_grad():
                best, save_best_flag, acc = evaluate(best)
            # save best
            if save_best_flag:
                save_student_model_flag = (
                    True if "Distillation" in cfg.MODEL.framework else False
                )
                if cfg.get("Global") is not None:
                    metric_info = {"metric": acc, "epoch": epoch}
                    prefix = "best_model"
                    model_path = osp.join(output_dir, prefix)
                    _mkdir_if_not_exist(model_path, logger)
                    model_path = osp.join(model_path, prefix)
                    _mkdir_if_not_exist(model_path, logger)
                    save(
                        model.state_dict(),
                        os.path.join(model_path, "model.pdparams"),
                        save_student_model=save_student_model_flag,
                    )

                    save(optimizer.state_dict(), model_path + ".pdopt")
                    save(
                        model.state_dict(),
                        model_path + ".pdparams",
                        save_student_model=save_student_model_flag,
                    )
                    save(metric_info, model_path + ".pdstates")

                    if uniform_output_enabled:
                        save_path = os.path.join(output_dir, prefix, "inference")
                        export(
                            cfg,
                            model,
                            save_path=save_path,
                            uniform_output_enabled=uniform_output_enabled,
                            logger=logger,
                        )

                        update_train_results(cfg, prefix, metric_info, ema=None)
                        save_model_info(metric_info, output_dir, prefix)
                else:
                    save(
                        optimizer.state_dict(),
                        osp.join(output_dir, model_name + "_best.pdopt"),
                    )
                    save_student_model_flag = (
                        True if "Distillation" in cfg.MODEL.framework else False
                    )
                    save(
                        model.state_dict(),
                        osp.join(output_dir, model_name + "_best.pdparams"),
                        save_student_model=save_student_model_flag,
                    )

                if model_name == "AttentionLstm":
                    logger.info(f"Already save the best model (hit_at_one){best}")
                elif cfg.MODEL.framework == "FastRCNN":
                    logger.info(
                        f"Already save the best model (mAP@0.5IOU){int(best * 10000) / 10000}"
                    )
                elif cfg.MODEL.framework == "DepthEstimator":
                    logger.info(
                        f"Already save the best model (rmse){int(best * 10000) / 10000}"
                    )
                elif cfg.MODEL.framework in ["MSTCN", "ASRF"]:
                    logger.info(
                        f"Already save the best model (F1@0.50){int(best * 10000) / 10000}"
                    )
                elif cfg.MODEL.framework in ["YOWOLocalizer"]:
                    logger.info(
                        f"Already save the best model (fsocre){int(best * 10000) / 10000}"
                    )
                else:
                    logger.info(
                        f"Already save the best model (top1 acc){int(best * 10000) / 10000}"
                    )

        # 10. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            if cfg.get("Global") is not None:
                metric_info = {"metric": acc, "epoch": epoch}
                prefix = "epoch_{}".format(epoch)
                model_path = osp.join(output_dir, prefix)
                _mkdir_if_not_exist(model_path, logger)
                model_path = osp.join(model_path, prefix)
                save(optimizer.state_dict(), model_path + ".pdopt")
                save(model.state_dict(), model_path + ".pdparams")
                save(metric_info, model_path + ".pdstates")

                if uniform_output_enabled:
                    save_path = os.path.join(output_dir, prefix, "inference")
                    export(
                        cfg,
                        model,
                        save_path=save_path,
                        uniform_output_enabled=uniform_output_enabled,
                        logger=logger,
                    )
                    update_train_results(
                        cfg,
                        prefix,
                        metric_info,
                        done_flag=epoch + 1 == cfg["epochs"],
                        ema=None,
                    )
                    save_model_info(metric_info, output_dir, prefix)
            else:
                save(
                    optimizer.state_dict(),
                    osp.join(output_dir, model_name + f"_epoch_{epoch + 1:05d}.pdopt"),
                )
                save(
                    model.state_dict(),
                    osp.join(
                        output_dir, model_name + f"_epoch_{epoch + 1:05d}.pdparams"
                    ),
                )
        if cfg.get("Global") is not None:
            metric_info = {"metric": acc, "epoch": epoch}
            prefix = "latest"
            model_path = osp.join(output_dir, prefix)
            _mkdir_if_not_exist(model_path, logger)
            model_path = osp.join(model_path, prefix)
            save(optimizer.state_dict(), model_path + ".pdopt")
            save(model.state_dict(), model_path + ".pdparams")
            save(metric_info, model_path + ".pdstates")

            if uniform_output_enabled:
                save_path = os.path.join(output_dir, prefix, "inference")
                export(
                    cfg,
                    model,
                    save_path=save_path,
                    uniform_output_enabled=uniform_output_enabled,
                    logger=logger,
                )
                save_model_info(metric_info, output_dir, prefix)

    logger.info(f"training {model_name} finished")


def export(
    cfg,
    model,
    save_path=None,
    uniform_output_enabled=False,
    ema_module=None,
    logger=None,
):
    assert uniform_output_enabled
    if paddle.distributed.get_rank() != 0:
        return

    if hasattr(model, "_layers"):
        model = copy.deepcopy(model._layers)
    else:
        model = copy.deepcopy(model)

    model.eval()
    for layer in model.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()

    if not save_path:
        save_path = os.path.join(cfg.Global.save_inference_dir, "inference")
    else:
        save_path = os.path.join(save_path, "inference")

    model_name = cfg.model_name
    input_spec = get_input_spec(cfg.INFERENCE, model_name)
    model = to_static(model, input_spec=input_spec)
    paddle.jit.save(model, save_path)
    if cfg["Global"].get("export_for_fd", False) or uniform_output_enabled:
        dst_path = os.path.join(os.path.dirname(save_path), "inference.yml")
        target_size = cfg["INFERENCE"]["target_size"]
        infer_shape = [3, target_size, target_size]
        dump_infer_config(cfg, dst_path, infer_shape, logger)
    logger.info(
        f'Export succeeded! The inference model exported has been saved in "{save_path}".'
    )


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping("tag:yaml.org,2002:map", dict_data.items())


def setup_orderdict():
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def dump_infer_config(inference_config, path, infer_shape, logger):
    setup_orderdict()
    infer_cfg = OrderedDict()
    config = inference_config

    if config["Global"].get("algorithm", None):
        infer_cfg["Global"] = {"model_name": config["Global"]["algorithm"]}

    if config["Infer"].get("transforms"):
        transforms = config["Infer"]["transforms"]
    else:
        logger.error("This config does not support dump transform config!")

    # Configuration required config for high-performance inference.
    if config["Global"].get("uniform_output_enabled"):
        infer_shape_with_batch = [
            [1] + infer_shape,
            [1] + infer_shape,
            [8] + infer_shape,
        ]

        dynamic_shapes = {"x": infer_shape_with_batch}

        backend_keys = ["paddle_infer", "tensorrt"]
        hpi_config = {
            "backend_configs": {
                key: {
                    (
                        "dynamic_shapes" if key == "tensorrt" else "trt_dynamic_shapes"
                    ): dynamic_shapes
                }
                for key in backend_keys
            }
        }

        infer_cfg["Hpi"] = hpi_config
    for transform in transforms:
        if "NormalizeImage" in transform:
            transform["NormalizeImage"]["channel_num"] = 3
            scale_str = transform["NormalizeImage"]["scale"]
            numerator, denominator = scale_str.split("/")
            numerator, denominator = float(numerator), float(denominator)
            transform["NormalizeImage"]["scale"] = float(numerator / denominator)
    infer_cfg["PreProcess"] = {
        "transform_ops": [
            infer_preprocess
            for infer_preprocess in transforms
            if "DecodeImage" not in infer_preprocess
        ]
    }
    if config.get("Infer"):
        if config["Infer"].get("PostProcess"):
            if config["Global"].get("algorithm") == "YOWO":
                infer_cfg["PostProcess"] = {
                     "transform_ops": [
                        post_op for post_op in config["Infer"].get("PostProcess")
                     ]
                }
                infer_cfg["label_list"] = config.get("label_list")
            else:
                postprocess_dict = copy.deepcopy(dict(config["Infer"]["PostProcess"]))
                with open(postprocess_dict["class_id_map_file"], "r", encoding="utf-8") as f:
                    label_id_maps = f.readlines()
                label_names = []
                for line in label_id_maps:
                    line = line.strip().split(" ", 1)
                    label_names.append(line[1:][0])

                postprocess_name = postprocess_dict.get("name", None)
                postprocess_dict.pop("class_id_map_file")
                postprocess_dict.pop("name")
                dic = OrderedDict()
                for item in postprocess_dict.items():
                    dic[item[0]] = item[1]
                dic["label_list"] = label_names

                if postprocess_name:
                    infer_cfg["PostProcess"] = {postprocess_name: dic}
                else:
                    raise ValueError("PostProcess name is not specified")
        else:
            infer_cfg["PostProcess"] = {"NormalizeFeatures": None}
    with open(path, "w") as f:
        yaml.dump(infer_cfg, f)
    logger.info("Export inference config file to {}".format(os.path.join(path)))
