### 🚀 Getting Started

Install the required Python packages with ```pip3 install -r requirements.txt```.
---

### 🧪 Experiments

To reproduce the experiments, follow these steps:

#### 1. Download and preprocess the data

```bash
mkdir -p tmp/ && python3 preprocess_public_data.py
```

Make sure you have correctly set up your OpenAI API credentials if you choose this option.


#### 2. Run the model
fusion_mode=weighted_sum, batchsize=128, scheduler=constantLR
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/amzn23_office/hstu-blair-weightedsum-batchsize128-constantlr.gin --master_port=12345
```

fusion_mode=weighted_sum, batchsize=512, scheduler=cosineannealingLR
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/amzn23_office/hstu-blair-weightedsum-batchsize512-cosinelr.gin --master_port=12345
```
