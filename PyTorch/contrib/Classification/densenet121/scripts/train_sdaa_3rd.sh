cd ..
pip install -r requirements.txt
# torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a densenet121 -b 64
# torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a densenet121 -b 64 >> ./scripts/train_sdaa_3rd_densenet121.log 2>&1
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29502 main.py /data/datasets/imagenet -a densenet121 -b 64