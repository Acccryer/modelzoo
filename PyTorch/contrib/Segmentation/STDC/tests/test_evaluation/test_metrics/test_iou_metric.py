# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import shutil
from unittest import TestCase

import numpy as np
import torch
import torch_sdaa
from mmengine.structures import PixelData

from mmseg.evaluation import IoUMetric
from mmseg.structures import SegDataSample


class TestIoUMetric(TestCase):

    def _demo_mm_inputs(self,
                        batch_size=2,
                        image_shapes=(3, 64, 64),
                        num_classes=5):
        """Create a superset of inputs needed to run test or train batches.

        Args:
            batch_size (int): batch size. Default to 2.
            image_shapes (List[tuple], Optional): image shape.
                Default to (3, 64, 64)
            num_classes (int): number of different classes.
                Default to 5.
        """
        if isinstance(image_shapes, list):
            assert len(image_shapes) == batch_size
        else:
            image_shapes = [image_shapes] * batch_size

        data_samples = []
        for idx in range(batch_size):
            image_shape = image_shapes[idx]
            _, h, w = image_shape

            data_sample = SegDataSample()
            gt_semantic_seg = np.random.randint(
                0, num_classes, (1, h, w), dtype=np.uint8)
            gt_semantic_seg = torch.LongTensor(gt_semantic_seg)
            gt_sem_seg_data = dict(data=gt_semantic_seg)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

            data_samples.append(data_sample.to_dict())

        return data_samples

    def _demo_mm_model_output(self,
                              data_samples,
                              batch_size=2,
                              image_shapes=(3, 64, 64),
                              num_classes=5):

        _, h, w = image_shapes

        for data_sample in data_samples:
            data_sample['seg_logits'] = dict(
                data=torch.randn(num_classes, h, w))
            data_sample['pred_sem_seg'] = dict(
                data=torch.randint(0, num_classes, (1, h, w)))
            data_sample[
                'img_path'] = 'tests/data/pseudo_dataset/imgs/00000_img.jpg'
        return data_samples

    def test_evaluate(self):
        """Test using the metric in the same way as Evalutor."""

        data_samples = self._demo_mm_inputs()
        data_samples = self._demo_mm_model_output(data_samples)

        iou_metric = IoUMetric(iou_metrics=['mIoU'])
        iou_metric.dataset_meta = dict(
            classes=['wall', 'building', 'sky', 'floor', 'tree'],
            label_map=dict(),
            reduce_zero_label=False)
        iou_metric.process([0] * len(data_samples), data_samples)
        res = iou_metric.evaluate(2)
        self.assertIsInstance(res, dict)

        # test save segment file in output_dir
        iou_metric = IoUMetric(iou_metrics=['mIoU'], output_dir='tmp')
        iou_metric.dataset_meta = dict(
            classes=['wall', 'building', 'sky', 'floor', 'tree'],
            label_map=dict(),
            reduce_zero_label=False)
        iou_metric.process([0] * len(data_samples), data_samples)
        assert osp.exists('tmp')
        assert osp.isfile('tmp/00000_img.png')
        shutil.rmtree('tmp')

        # test format_only
        iou_metric = IoUMetric(
            iou_metrics=['mIoU'], output_dir='tmp', format_only=True)
        iou_metric.dataset_meta = dict(
            classes=['wall', 'building', 'sky', 'floor', 'tree'],
            label_map=dict(),
            reduce_zero_label=False)
        iou_metric.process([0] * len(data_samples), data_samples)
        assert iou_metric.results == []
        assert osp.exists('tmp')
        assert osp.isfile('tmp/00000_img.png')
        shutil.rmtree('tmp')
