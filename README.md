## Implementing Recommendation Algorithms using Generative Models 
#### This branch corresponds to the preliminary experiment code.

This branch only implements the sum, gated, MLP, and ResNet MLP fusion methods mentioned in the paper. 
The Hadamard method was confirmed to be ineffective and will not be used in the main experiments.

The model was trained on a single RTX 4090 GPU.
A single training run of 100 epochs takes approximately 7 hours.
Please refer to requirements.txt for the environment configuration.

Before training, use the following command to prepare the data and BLair embedding.
```bash 
mkdir -p tmp/ && python3 preprocess_public_data.py
```

To start the training, please run the following command.
```bash 
python main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair.gin --master_port=12345
```

To modify the running hyperparameters, please adjust the configuration file listed below:

configs/amzn23_office/hstu-sampled-softmax-n512-blair.gin

The available options for the train_fn.fusion_mode parameter are: `sum`, `concat_mlp`, `gated`, `resnet_mlp`, and `no_fusion` (in this mode, only the item ID embedding will be used).