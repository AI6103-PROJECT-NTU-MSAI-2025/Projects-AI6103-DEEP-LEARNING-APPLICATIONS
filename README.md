# Branch Notes
In this branch, batch_size is set to 512 and a cosine learning rate scheduler is enabled for training.


## 1. Download and preprocess the data

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

## 2. Run experiments
```sh
# baseline
python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-baseline.gin --master_port=12345

# blair
python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair.gin --master_port=12345

# blair-concat(MLP)
python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair-concat.gin --master_port=12345

# blair-gated
python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair-gated.gin --master_port=12345

# blair-reset
python3 main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair-resnet.gin --master_port=12345
```

