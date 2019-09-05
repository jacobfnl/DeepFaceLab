import argparse
import multiprocessing
from utils import os_utils
from mainscripts import Trainer

CONFIG_ARGS = {
    'training_data_src_dir'  : '../DeepFaceLab-workspace/carrey',
    'training_data_dst_dir'  : '../DeepFaceLab-workspace/reeves',
    'pretraining_data_dir'   : '',
    'model_path'             : '../DeepFaceLab-workspace/model',
    'model_name'             : 'SAE',
    'no_preview'             : True,
    'debug'                  : False,
    'execute_program'        : [],
}

CONFIG_DEVICE_ARGS = {
    'cpu_only'               : True,
    'force_gpu_idx'          : -1,
}


def main():
    multiprocessing.set_start_method("spawn")
    os_utils.set_process_lowest_prio()
    Trainer.main(CONFIG_ARGS, CONFIG_DEVICE_ARGS)


if __name__ == "__main__":
    main()
