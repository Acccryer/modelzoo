# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import onnx

import dump
from dump.sdaa_tvm_executor import SdaaTvmExecutor


def main():
    # 模型路径
    model_name = "model_sample.onnx"
    passes_path = "ocr_passes.py"

    # 加载 onnx 模型
    model = onnx.load(model_name)

    # 初始化 DumpInfo，添加模型和随机数据的相关信息
    dump_info = dump.DumpInfo(root_dir="model_sample_4")
    dump_info.set_meta_info({
        "model_name": model_name,
        "model_dtype": "float16",
        "default_seed": 42,
        "default_float_min": 0.0,
        "default_float_max": 1.0,
        "default_int_min": 0,
        "default_int_max": 100,
    })

    # sdaa 相关设置
    load_model_configs = {}
    load_model_configs["dtype"] = "float16"
    load_model_configs["target"] = "sdaa --libs=tecodnn,tecoblas"
    load_model_configs["device_type"] = "sdaa"
    load_model_configs["passes"] = passes_path
    load_model_configs["disabled_pass"] = ["SimplifyInference"]
    load_model_configs["opt_level"] = 3
    load_model_configs["build_config"] = None

    execute_configs = {}
    execute_configs["use_device_id"] = None

    # 初始化 Pipeline
    pipeline = dump.BasePipeline(model,
                                 SdaaTvmExecutor(),
                                 dump_info,
                                 load_model_configs=load_model_configs,
                                 execute_configs=execute_configs)

    # 设置输入，使用DumpInfo中的随机数据信息产生输入数据
    pipeline.set_input_data()

    # BasePipeline中提供的dump模式，将全部node的输出tensor依次计算结果并存储
    pipeline.dump_full()


if __name__ == "__main__":
    main()
