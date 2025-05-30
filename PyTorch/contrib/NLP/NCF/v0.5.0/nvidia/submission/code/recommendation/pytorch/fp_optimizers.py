# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import torch
import torch_sdaa

class Fp16Optimizer:

    def __init__(self, fp16_model, loss_scale=8192.0):
        print('Initializing fp16 optimizer')
        self.initialize_model(fp16_model)
        self.loss_scale = loss_scale

    def initialize_model(self, model):
        print('Reset fp16 grad')
        self.fp16_model = model
        for param in self.fp16_model.parameters():
            param.grad = None

        print('Initializing fp32 clone weights')
        self.fp32_params = [param.clone().type(torch.sdaa.FloatTensor).detach()
                            for param in model.parameters()]
        for param in self.fp32_params:
            param.requires_grad = True

    def step(self, loss, optimizer):
        loss *= self.loss_scale
        for p in self.fp16_model.parameters():
            p.grad = None
        loss.backward()

        optimizer.step(grads=[p.grad for p in 
            self.fp16_model.parameters()], output_params=self.fp16_model.parameters(), scale=self.loss_scale)

class Fp32Optimizer:
    def __init__(self, model, grad_clip=None):
        print('Initializing fp32 optimizer')
        self.initialize_model(model)
        self.grad_clip = grad_clip

    def initialize_model(self, model):
        self.model = model
        self.model.zero_grad()

    def step(self, loss, optimizer):
        loss.backward()
        if self.grad_clip != float('inf'):
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()
        self.model.zero_grad()
