import os
import shutil
import multiprocessing
import mysql.connector
from mainscripts import Trainer
from mainscripts.Util import recover_original_aligned_filename
from utils import Path_utils, os_utils
from interact import interact as io
from pathlib import Path
from utils.init_workspace import init_workspace
import argparse

workspace, src_aligned, dst_aligned, workspace_model = init_workspace()
data_src = os.path.abspath(os.path.join(src_aligned, os.pardir))
data_dst = os.path.abspath(os.path.join(dst_aligned, os.pardir))


character_bucket = '/media/warriordata/character_buckets'
live_faces = '/media/warriordata/live_faces'


def open_db_connection():
    DB_USER = os.getenv('DB_USER')
    DB_PASS = os.getenv('DB_PASS')
    DB_ADDRESS = os.getenv('DB_ADDRESS')
    DB_DB = os.getenv('DB_DB')
    config = {
        'user': DB_USER,
        'password': DB_PASS,
        'host': DB_ADDRESS,
        'database': DB_DB,
        'raise_on_warnings': True
    }
    return mysql.connector.connect(**config)


class fixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def process_train(training_data_src_dir,training_data_dst_dir, model_dir, force_gpu_idx, debug_dir=None):
    os_utils.set_process_lowest_prio()
    args = {'training_data_src_dir': training_data_src_dir,
            'training_data_dst_dir': training_data_dst_dir,
            'pretraining_data_dir': None,
            'model_path': model_dir,
            'model_name': 'SAEHD',
            'no_preview': False,
            'debug': debug_dir,
            'flask_preview': True,
            'execute_programs': [],
            }
    device_args = {'cpu_only': False,
                   'force_gpu_idx': force_gpu_idx,
                   'use_fp16': False
                   }
    from mainscripts import Trainer
    Trainer.main(args, device_args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()

    def process_args(args):
        live_person_uuid = args.live_person
        link_id = args.link_id
        gpu_idx = args.gpu_idx
        training_data_dst_dir = os.path.join(character_bucket, link_id, 'chips')
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace/model_' + live_person_uuid)
        db = open_db_connection()
        query = f"SELECT COPYRIGHT FROM inconode where CONTENT_TYPE='extracted' AND DISTRIBUTION='{live_person_uuid}'"
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        yymmdd = ''
        if len(result):
            yymmdd = result[0][0]
        if yymmdd == '':
            print(f"\nError, could not find person: {live_person_uuid} \nPerhaps they've not been extracted, "
                  f"or have a different CONTENT_TYPE code in the database.")
            exit(-1)
        training_data_src_dir = os.path.join(live_faces, yymmdd, live_person_uuid, 'chips_224')
        process_train(training_data_src_dir=training_data_src_dir, training_data_dst_dir=training_data_dst_dir,
                      model_dir=model_dir, force_gpu_idx=gpu_idx)

    parser.add_argument('-person', required=True, dest="live_person",
                        help="live_person uuid")
    parser.add_argument('-link-id', required=True, dest="link_id",
                        help="film character link_id")
    parser.add_argument('-gpu', type=int, dest="gpu_idx", default=-1, help="Force to choose this GPU idx.")

    parser.set_defaults(func=process_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
