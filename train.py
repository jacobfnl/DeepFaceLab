#!/usr/bin/env python3

import multiprocessing
from utils import os_utils
from mainscripts import Trainer

CONFIG_ARGS = {
    'training_data_src_dir'  : '../DeepFaceLab-workspace/carrey',
    'training_data_dst_dir'  : '../DeepFaceLab-workspace/reeves',
    'pretraining_data_dir'   : '',
    'model_path'             : '../DeepFaceLab-workspace/model',
    'model_name'             : 'SAE',
    'no_preview'             : False,
    'debug'                  : False,
    'execute_program'        : [],
}

CONFIG_DEVICE_ARGS = {
    'cpu_only'               : True,
    'force_gpu_idx'          : -1,
}

CONFIG_MODEL_BASE_OPTIONS = {
    'autobackup'             : False,
    'write_preview_history'  : True,
    'target_iter'            : 0,
    'batch_size'             : 8,
    'batch_cap'              : 4,
    'ping_pong'              : False,
    'paddle'                 : 'ping',
    'ping_pong_iter'         : 1000,
    'sort_by_yaw'            : False,
    'random_flip'            : True,
    'src_scale_mod'          : 0,
    'choose_preview_history' : False,
}

CONFIG_SAE_MODEL_OPTIONS = {
    'resolution'             : 16,
    'face_type'              : 'f',
    'learn_mask'             : True,
    'optimizer_mode'         : 1,
    'archi'                  : 'df',
    'ae_dims'                : 64,
    'e_ch_dims'              : 4,
    'd_ch_dims'              : 2,
    'multiscale_decoder'     : False,
    'ca_weights'             : True,
    'pixel_loss'             : False,
    'face_style_power'       : 0.0,
    'bg_style_power'         : 0.0,
    'default_apply_random_ct': 5,
    'clipgrad'               : False,
    'pretrain'               : False,
}


def main():
    multiprocessing.set_start_method("spawn")
    os_utils.set_process_lowest_prio()
    Trainer.main(CONFIG_ARGS, CONFIG_DEVICE_ARGS)


if __name__ == "__main__":
    main()
