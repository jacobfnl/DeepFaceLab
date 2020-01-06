import time
import os
import shutil
import mysql.connector
import logging
from utils import Path_utils, os_utils
from pathlib import Path
import multiprocessing

from mainscripts import VideoEd, Extractor


logging.basicConfig(filename='log_02_warriors_extractor.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
logger = logging.getLogger('warriors_extractor')


def current_dir_path():
    return os.path.dirname(os.path.abspath(__file__))


def warriors_path():
    return '/media/warriordata/live_faces/'


def video_path():
    return os.path.abspath(os.path.join(png_path(), 'videos'))


def png_path():
    return os.path.abspath(os.path.join(current_dir_path(), '..', 'processing_pngs'))


def chips_224_path():
    return os.path.join(png_path(), 'chips_224')


def verify_paths():
    dir_chip224 = chips_224_path()
    if not os.path.exists(dir_chip224):
        logger.info("Chips path does not exist. Creating Directories:\n{}".format(dir_chip224))
        os.makedirs(dir_chip224)


def update_process_file():
    print(update_process_file.__name__)
    # process_file contains timestamp and stage of process
    # we can then use this to determine if any part hangs.
    # TODO: define acceptable process stage durations.
    return True


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


def check_database():
    # print(check_database.__name__)
    cursor = db.cursor()
    # fetch one result
    query = "SELECT ID, BORN, URL, DISTRIBUTION, COPYRIGHT FROM inconode " \
            "where CONTENT_TYPE ='todo' order by id"
    cursor.execute(query)
    result = cursor.fetchall()
    if len(result):
        key_id, created, url, uuid, ymd = result[0]
        return key_id, created, url, uuid, ymd
    return None, None, None, None, None


def check_file_exists(_url, _uuid):
    path = Path(_url)
    stem = path.stem
    ext = path.suffix
    local_uuid_path = os.path.join(video_path(), _uuid)
    if not os.path.exists(local_uuid_path):
        logger.info("path does not exist. {} ".format(local_uuid_path))
        os.makedirs(local_uuid_path)
    local_file = os.path.join(local_uuid_path, stem+ext)
    logger.info("local_path: {}".format(local_file ))
    if not os.path.exists(local_file ):
        logger.info("copy from absolute path")
        shutil.copyfile(_url, local_file)
    else:
        logger.info("path exists.")
    return local_file, local_uuid_path


def frames_from_video(_video_file, _uuid_path):
    logger.info("processing video...")
    begin_time = time.time()
    VideoEd.extract_video(_video_file, _uuid_path, output_ext='png', fps=0)
    delta_time = time.time()
    logger.info("frames processed in: {:.4f}s".format(delta_time-begin_time))


def extract_from_frames(_input_dir, _uuid):
    print(extract_from_frames.__name__)
    # read pngs from local storage
    logger.info("starting Extractor...")
    _output_dir = os.path.join(chips_224_path(), _uuid)
    _debug_dir = os.path.join(_output_dir, 'debug', 'image_sequence')

    begin_time = time.time()
    Extractor.main(_input_dir, _output_dir, _debug_dir, 's3fd', image_size=224, face_type='full_face',
                   max_faces_from_image=1, character_number=0, gamma=1.0)
    delta_time = time.time()
    logger.info("extraction finished in: {:.4f}s".format(delta_time-begin_time))
    return _output_dir, _debug_dir


def create_debug_video(_debug_dir, _local_vid):
    print("video from debug dir")
    output_file_name = Path(_local_vid).stem + '.mp4'
    debug_video_dir = Path(_debug_dir).parent
    output_file = os.path.join(debug_video_dir, output_file_name)
    begin_time = time.time()
    VideoEd.video_from_sequence(_debug_dir, output_file, _local_vid, ext='jpg', bitrate=4, lossless=False)
    delta_time = time.time()
    logger.info("debug encode finished in: {:.4f}s".format(delta_time - begin_time))
    # remove the debug jpgs, just keep the mp4.
    shutil.rmtree(_debug_dir)


def save_chips(_nas_video_url, _local_chips, _local_vid, _uuid, _ymd):
    print(save_chips.__name__)
    begin_time = time.time()
    if not os.path.exists(_nas_video_url):
        logger.info("file is not on NAS... uploading.")
        shutil.copyfile(_local_vid, _nas_video_url)
    dst_root = os.path.join(warriors_path(), _ymd, _uuid)
    dst_chips = os.path.join(dst_root, 'chips_224')
    shutil.copytree(_local_chips, dst_chips)
    delta_time = time.time()
    logger.info("Files Transfer: {:.4f}s".format(delta_time - begin_time))


def find_frontal_image():
    # TODO: do we need a frontal image for ID purposes?
    print(find_frontal_image.__name__)


def send_to_clarifai():
    # TODO: do we send this from here? Or is it earlier in the process.
    print(send_to_clarifai.__name__)


def database_it(_id):
    print(database_it.__name__)
    logger.info("updating database")
    _now = time.strftime('%Y-%m-%d %H:%M:%S')
    _query = "UPDATE inconode SET `UPDATE`='{}', CONTENT_TYPE='extracted' WHERE ID={}".format(_now, _id)
    _cursor = db.cursor()
    _cursor.execute(_query)
    db.commit()
    result = _cursor.close()


def delete_excess(_uuid):
    _chips = os.path.join(chips_224_path(), _uuid)
    _video = os.path.join(video_path(), _uuid)
    shutil.rmtree(_chips)
    shutil.rmtree(_video)


cycle = 0

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    os_utils.set_process_lowest_prio()
    verify_paths()
    logger.info("starting observations")
    db = open_db_connection()
    while True:
        time.sleep(1)
        cycle += 1
        logger.debug("cycle: {}".format(cycle))
        key_id, created, url, uuid, ymd = check_database()
        if key_id is None:
            continue

        logger.info("processing file: id: {} created: {}, location: {}".format(key_id, created, url))
        loop_begin = time.time()
        local_vid, uuid_path = check_file_exists(_url=url, _uuid=uuid)
        frames_from_video(local_vid, _uuid_path=uuid_path)
        output_chips, debug_dir = extract_from_frames(_input_dir=uuid_path, _uuid=uuid)
        create_debug_video(_debug_dir=debug_dir, _local_vid=local_vid)
        save_chips(_nas_video_url=url, _local_chips=output_chips, _local_vid=local_vid, _uuid=uuid, _ymd=ymd)
        database_it(key_id)
        delete_excess(_uuid=uuid)
        loop_delta = time.time()
        logger.info("Finished processing. Total time: {:.4f}s".format(loop_delta - loop_begin))

    # end of line.
    db.close()
