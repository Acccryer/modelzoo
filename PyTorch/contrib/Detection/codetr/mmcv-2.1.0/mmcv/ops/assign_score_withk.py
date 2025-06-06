from typing import Tuple

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['assign_score_withk_forward', 'assign_score_withk_backward'])


class AssignScoreWithK(Function):
    r"""Perform weighted sum to generate output features according to scores.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/paconv_lib/src/gpu>`_.

    This is a memory-efficient SDAA implementation of assign_scores operation,
    which first transform all point features with weight bank, then assemble
    neighbor features with ``knn_idx`` and perform weighted sum of ``scores``.

    See the `paper <https://arxiv.org/pdf/2103.14635.pdf>`_ appendix Sec. D for
        more detailed descriptions.

    Note:
        This implementation assumes using ``neighbor`` kernel input, which is
            (point_features - center_features, point_features).
        See https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/model/
        pointnet2/paconv.py#L128 for more details.
    """

    @staticmethod
    def forward(ctx,
                scores: torch.Tensor,
                point_features: torch.Tensor,
                center_features: torch.Tensor,
                knn_idx: torch.Tensor,
                aggregate: str = 'sum') -> torch.Tensor:
        """
        Args:
            scores (torch.Tensor): (B, npoint, K, M), predicted scores to
                aggregate weight matrices in the weight bank.
                ``npoint`` is the number of sampled centers.
                ``K`` is the number of queried neighbors.
                ``M`` is the number of weight matrices in the weight bank.
            point_features (torch.Tensor): (B, N, M, out_dim)
                Pre-computed point features to be aggregated.
            center_features (torch.Tensor): (B, N, M, out_dim)
                Pre-computed center features to be aggregated.
            knn_idx (torch.Tensor): (B, npoint, K), index of sampled kNN.
                We assume the first idx in each row is the idx of the center.
            aggregate (str, optional): Aggregation method.
                Can be 'sum', 'avg' or 'max'. Defaults: 'sum'.

        Returns:
            torch.Tensor: (B, out_dim, npoint, K), the aggregated features.
        """
        agg = {'sum': 0, 'avg': 1, 'max': 2}

        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()

        output = point_features.new_zeros((B, out_dim, npoint, K))
        ext_module.assign_score_withk_forward(
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            output,
            B=B,
            N0=N,
            N1=npoint,
            M=M,
            K=K,
            O=out_dim,
            aggregate=agg[aggregate])

        ctx.save_for_backward(output, point_features, center_features, scores,
                              knn_idx)
        ctx.agg = agg[aggregate]

        return output

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Args:
            grad_out (torch.Tensor): (B, out_dim, npoint, K)

        Returns:
            tuple[torch.Tensor]: A tuple contains five elements. The first one
            is the gradient of ``scores`` whose shape is (B, npoint, K, M). The
            second is the gradient of ``point_features`` whose shape is
            (B, N, M, out_dim). The third is the gradient of
            ``center_features`` with the shape of (B, N, M, out_dim). The last
            two are ``None``.
        """
        _, point_features, center_features, scores, knn_idx = ctx.saved_tensors

        agg = ctx.agg

        B, N, M, out_dim = point_features.size()
        _, npoint, K, _ = scores.size()

        grad_point_features = point_features.new_zeros(point_features.shape)
        grad_center_features = center_features.new_zeros(center_features.shape)
        grad_scores = scores.new_zeros(scores.shape)

        ext_module.assign_score_withk_backward(
            grad_out.contiguous(),
            point_features.contiguous(),
            center_features.contiguous(),
            scores.contiguous(),
            knn_idx.contiguous(),
            grad_point_features,
            grad_center_features,
            grad_scores,
            B=B,
            N0=N,
            N1=npoint,
            M=M,
            K=K,
            O=out_dim,
            aggregate=agg)

        return grad_scores, grad_point_features, \
            grad_center_features, None, None


assign_score_withk = AssignScoreWithK.apply
