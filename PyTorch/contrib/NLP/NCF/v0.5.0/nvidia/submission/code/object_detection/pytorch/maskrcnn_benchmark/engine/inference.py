# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

import datetime
import logging
import tempfile
import time
from collections import OrderedDict

import torch
import torch_sdaa

from tqdm import tqdm

from ..structures.bounding_box import BoxList
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize
from maskrcnn_benchmark.utils.mlperf_logger import print_mlperf
from mlperf_compliance import mlperf_log


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

import os
from multiprocessing import Pool

def prepare_for_coco_segmentation(predictions, dataset, rank):
    from maskrcnn_benchmark.modeling.post_processors.mask_rcnn import Masker
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    offset = rank*len(predictions)
    for image_id, prediction in tqdm(enumerate(predictions)):

        image_id = image_id+offset
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction.bbox) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask1")
        # t = time.time()
        masks = masker(masks, prediction)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0
    from maskrcnn_benchmark.modeling.box_ops import boxes_iou

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if gt_boxes.bbox.shape[0] == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += gt_boxes.bbox.shape[0]

        if gt_boxes.bbox.shape[0] == 0:
            continue

        if len(prediction.bbox) == 0:
            continue

        if limit is not None and prediction.bbox.shape[0] > limit:
            prediction = prediction[:limit]

        overlaps = boxes_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(gt_boxes.bbox.shape[0])
        for j in range(min(prediction.bbox.shape[0], gt_boxes.bbox.shape[0])):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }

def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    # torch.save(coco_eval, join(args.save, 'computed_results.pth'))
    res = coco_eval.summarize()
    # return the required metric
    return [res, coco_eval.stats[0]]


def _is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not _is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)


def inference(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset

    print_mlperf(key=mlperf_log.EVAL_SIZE, value=len(dataset))

    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not _is_main_process():
        return

    for p in predictions:
        p.add_field('mask1', p.get_field('mask').float())

    if box_only:
        logger.info("Evaluating bbox proposals")
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = evaluate_box_proposals(
                    predictions, dataset, area=area, limit=limit
                )
                key = "AR{}@{:d}".format(suffix, limit)
                res.results["box_proposal"][key] = stats["ar"].item()
        # logger.info("AR@1000: {}".format(results["ar"].item()))
        logger.info(res)
        check_expected_results(res, expected_results, expected_results_sigma_tol)

        # return back result
        return res
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if "segm" in iou_types:
        logger.info("Preparing segm results")
        num_proc = 20
        local_len = len(predictions) // num_proc
        chunks = []
        for i in range(num_proc):
            chunks.append((predictions[i*local_len:(i+1)*local_len], dataset, i))
        def init(env):
            os.environ = env
        with Pool(num_proc, initializer=init, initargs=(os.environ.copy(),)) as p:
            t = p.starmap(prepare_for_coco_segmentation, chunks)

        tmp = []
        for i in range(num_proc):
            tmp += t[i]
        coco_results["segm"] = tmp

        #coco_results["segm"] = prepare_for_coco_segmentation(predictions, dataset)

    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")

    # Extracting only required metrics
    map_required = {}
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            [res, map] = evaluate_predictions_on_coco(
                dataset.coco, coco_results[iou_type], f.name, iou_type
            )
            map_required[iou_type] = map
            results.update(res)
    check_expected_results(res, expected_results, expected_results_sigma_tol)
    # returning results
    return map_required
