import logging
import os
from typing import List, Optional

import sys
import fbgemm_gpu  # noqa: F401, E402
import gin
import torch
import torch.multiprocessing as mp
from absl import app, flags
from generative_recommenders.research.trainer.train import train_fn
import time
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Hide excessive tensorflow debug messages and warning

def delete_flags(FLAGS, keys_to_delete: List[str]) -> None:
# redefine flags that might have been defined elsewhere
    keys = [key for key in FLAGS._flags()]
    for key in keys:
        if key in keys_to_delete:
            delattr(FLAGS, key)

delete_flags(flags.FLAGS, ["gin_config_file", "master_port"])
flags.DEFINE_string("gin_config_file", None, "Path to the config file.")
# Redefine string flag for the path to the Gin configuration file.
flags.DEFINE_integer("master_port", 12355, "Master port.")
# Redefine integer flag for the master port used by PyTorch Distributed.
FLAGS = flags.FLAGS
# Assign the globally available flags object to a local variable

def mp_train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    gin_config_file: Optional[str],
) -> None:
    # this is the training function

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Rank %(rank)d - %(levelname)s - %(message)s',
        stream=sys.stdout)
    # Explicitly reconfigure logging to ensure it works correctly in child processes.

    if gin_config_file is not None:
        logging.info(f"Rank {rank}: loading gin config from {gin_config_file}")
        gin.parse_config_file(gin_config_file)
        # Parse the configuration file using gin-config library.

    train_fn(rank, world_size, master_port)

def _main(argv) -> None:  # pyre-ignore [2]
    # Use PyTorch's multiprocessing spawn function to create worker processes to run train_fn
    world_size = torch.cuda.device_count()
    mp.set_start_method("forkserver")
    mp.spawn(
        mp_train_fn,
        args=(world_size, FLAGS.master_port, FLAGS.gin_config_file),
        nprocs=world_size,
        join=True,
    )

def main() -> None:
    app.run(_main)

if __name__ == "__main__":
    ## this is the entrance of training
    # python main.py --gin_config_file=configs/amzn23_office/hstu-sampled-softmax-n512-blair.gin --master_port=12345

    print(time)
    main()
    print(time)

