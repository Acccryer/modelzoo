#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


from ..model import ModelBase
from .tsn_res_model import TSN_ResNet

import logging
import paddle
import paddle_sdaa
import paddle.static as static
logger = logging.getLogger(__name__)

__all__ = ["TSN"]


class TSN(ModelBase):
    def __init__(self, name, cfg, mode='train', is_videotag=False):
        super(TSN, self).__init__(name, cfg, mode=mode)
        self.is_videotag = is_videotag
        self.get_config()

    def get_config(self):
        self.num_classes = self.get_config_from_sec('model', 'num_classes')
        self.seg_num = self.get_config_from_sec('model', 'seg_num')
        self.seglen = self.get_config_from_sec('model', 'seglen')
        self.image_mean = self.get_config_from_sec('model', 'image_mean')
        self.image_std = self.get_config_from_sec('model', 'image_std')
        self.num_layers = self.get_config_from_sec('model', 'num_layers')

        self.num_epochs = self.get_config_from_sec('train', 'epoch')
        self.total_videos = self.get_config_from_sec('train', 'total_videos')
        self.base_learning_rate = self.get_config_from_sec(
            'train', 'learning_rate')
        self.learning_rate_decay = self.get_config_from_sec(
            'train', 'learning_rate_decay')
        self.l2_weight_decay = self.get_config_from_sec('train',
                                                        'l2_weight_decay')
        self.momentum = self.get_config_from_sec('train', 'momentum')

        self.seg_num = self.get_config_from_sec(self.mode, 'seg_num',
                                                self.seg_num)
        self.target_size = self.get_config_from_sec(self.mode, 'target_size')
        self.batch_size = self.get_config_from_sec(self.mode, 'batch_size')

    def build_input(self, use_dataloader=True):
        image_shape = [3, self.target_size, self.target_size]
        image_shape[0] = image_shape[0] * self.seglen
        image_shape = [None, self.seg_num] + image_shape
        self.use_dataloader = use_dataloader

        image = static.data(name='image', shape=image_shape, dtype='float32')
        if self.mode != 'infer':
            label = static.data(name='label', shape=[None, 1], dtype='int64')
        else:
            label = None

        if use_dataloader:
            assert self.mode != 'infer', \
                        'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = paddle.io.DataLoader.from_generator(
                feed_list=[image, label], capacity=4, iterable=True)

        self.feature_input = [image]
        self.label_input = label

    def create_model_args(self):
        cfg = {}
        cfg['layers'] = self.num_layers
        cfg['class_dim'] = self.num_classes
        cfg['seg_num'] = self.seg_num
        return cfg

    def build_model(self):
        cfg = self.create_model_args()
        videomodel = TSN_ResNet(layers=cfg['layers'],
                                seg_num=cfg['seg_num'],
                                is_training=(self.mode == 'train'),
                                is_extractor=self.is_videotag)
        out = videomodel.net(input=self.feature_input[0],
                             class_dim=cfg['class_dim'])
        self.network_outputs = [out]

    def optimizer(self):
        assert self.mode == 'train', "optimizer only can be get in train mode"
        epoch_points = [self.num_epochs / 3, self.num_epochs * 2 / 3]
        total_videos = self.total_videos
        step = int(total_videos / self.batch_size + 1)
        bd = [e * step for e in epoch_points]
        base_lr = self.base_learning_rate
        lr_decay = self.learning_rate_decay
        lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
        l2_weight_decay = self.l2_weight_decay
        momentum = self.momentum
        optimizer = paddle.optimizer.Momentum(
            learning_rate=paddle.optimizer.lr.PiecewiseDecay(boundaries=bd,
                                                       values=lr),
            momentum=momentum,
            weight_decay=paddle.regularizer.L2Decay(coeff=l2_weight_decay))

        return optimizer

    def loss(self):
        assert self.mode != 'infer', "invalid loss calculationg in infer mode"
        cost = paddle.nn.functional.cross_entropy(input=self.network_outputs[0], \
                           label=self.label_input, ignore_index=-1)
        self.loss_ = paddle.mean(x=cost)
        return self.loss_

    def outputs(self):
        return self.network_outputs

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else self.feature_input + [
            self.label_input
        ]

    def fetches(self):
        if self.mode == 'train' or self.mode == 'valid':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'test':
            losses = self.loss()
            fetch_list = [losses, self.network_outputs[0], self.label_input]
        elif self.mode == 'infer':
            fetch_list = self.network_outputs
        else:
            raise NotImplementedError('mode {} not implemented'.format(
                self.mode))

        return fetch_list

    def pretrain_info(self):
        return None, None

    def weights_info(self):
        return None

    def load_pretrain_params(self, exe, pretrain, prog):
        def is_parameter(var):
            return isinstance(var, paddle.framework.Parameter)

        logger.info(
            "Load pretrain weights from {}, exclude fc layer.".format(pretrain))

        print("===pretrain===", pretrain)
        state_dict = paddle.static.load_program_state(pretrain)
        dict_keys = list(state_dict.keys())
        # remove fc layer when pretrain, because the number of classes in final fc may not match
        for name in dict_keys:
            if "fc_0" in name:
                del state_dict[name]
                print('Delete {} from pretrained parameters. Do not load it'.
                      format(name))
        paddle.static.set_program_state(prog, state_dict)
