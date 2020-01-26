import os
import shutil
import multiprocessing
import platform
from mainscripts import Extractor
from mainscripts.Util import recover_original_aligned_filename
from utils import Path_utils, os_utils, db_connection
from interact import interact as io
from pathlib import Path
import argparse
import subprocess

DATA_DST_ALIGNED = 'workspace/data_dst/aligned'
DEBUG_EXTRACTION_DIR = 'workspace/data_dst/debug_extraction'
live_faces = '/warriordata/live_faces'

class FixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def fix_manual_cache(input_path, source_path, character: int):
    recover_original_aligned_filename(input_path)
    source_chosen = os.path.join(source_path, 'chosen_frames')
    if os.path.exists(source_chosen):
        shutil.rmtree(source_chosen)
    os.makedirs(source_chosen)

    dir_to_fix = os.path.join(input_path, 'for_fixing')
    if os.path.exists(dir_to_fix):
        shutil.rmtree(dir_to_fix)
    os.makedirs(dir_to_fix)

    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'data_dst', 'debug_extraction')
    print("\ndebug_dir: {}\n".format(debug_dir))
    chosen_debug = os.path.join(dir_to_fix, '00-debug-frames')
    os.makedirs(chosen_debug)
    x = 0
    for filepath in io.progress_bar_generator(Path_utils.get_image_paths(input_path), "Processing New Manual Fix Order."):
        filepath = Path(filepath)
        # find the source image in source path.
        # example /home/cyrus/Documents/DFL/workspace/data_dst/warriors_src.086299_0.jpg
        file_parts = filepath.stem
        extention = filepath.suffix
        stem = file_parts
        print(stem)
        split_stem = stem.split('_')
        chosen_file = split_stem[0] + '_' + split_stem[1] + '.png'
        print(chosen_file)
        if not os.path.exists(os.path.join(source_path, chosen_file)):
            source_jpg = split_stem[0] + '_' + split_stem[1] + '.jpg'  # ¯\_(ツ)_/¯
            s_png = chosen_file
            if os.path.exists(os.path.join(source_path, source_jpg)):
                print("found jpg: {}".format(source_jpg))
                chosen_file = source_jpg
            else:
                print("Unable to find path: {}\nand not here either: {}".format(os.path.join(source_path, s_png),
                                                                                os.path.join(source_path, source_jpg)))
                exit()
        # copy chosen originals to source_chosen
        file_copy = filepath.stem + extention
        shutil.move(filepath, os.path.join(dir_to_fix, file_copy))

        chosen_file_path = Path(os.path.join(source_path, chosen_file))
        file_copy = chosen_file_path.stem + '_' + str(x) + '.png'
        shutil.copyfile(chosen_file_path, os.path.join(source_chosen, file_copy))

        # copy the debug frame to debug_dir
        chosen_file = chosen_file_path.stem + '.jpg'
        file_copy = chosen_file_path.stem + '_' + str(x) + '.jpg'
        if os.path.exists(os.path.join(debug_dir, chosen_file)):
            print("debug chosen file: {}".format(chosen_file))
            shutil.copyfile(os.path.join(debug_dir, chosen_file), os.path.join(chosen_debug, file_copy))
        else:
            print("debug file does not exist.")
        x += 1

    print("\n\nExamples of the images to fix have been placed in {}.\nPlease open those images for reference.".format(dir_to_fix))

    # now implement manual fix on the items.
    Extractor.main(source_chosen,
                   input_path,
                   None,
                   'manual',
                   False,
                   False,
                   1920,
                   face_type='full_face',
                   device_args={'multi_gpu': True},
                   character_number=character
                   )
    # remove the extra frame underscore
    for filepath in io.progress_bar_generator(Path_utils.get_image_paths(input_path), "Renaming files."):
        filepath = Path(filepath)
        parts = filepath.stem.split('_')
        file_name = parts[0] + '_' + parts[1] + '_' + parts[3] + filepath.suffix
        shutil.move(filepath, os.path.join(input_path, file_name))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    def fix_manual_alignments(arrrgs):
        fix_manual_cache(arrrgs.input, arrrgs.source, arrrgs.character)

    p = subparsers.add_parser("manual-fix", help="Manual fix of mis-alignments.")
    p.add_argument('-i', '--input', type=str, default=DATA_DST_ALIGNED,
                   help='Directory of aligned images you wish to re-do. Default is ' + DATA_DST_ALIGNED)
    p.add_argument('-s', '--source', type=str, default='workspace/data_dst',
                   help="Directory of Source Frames. Default is 'workspace/data_dst'")
    p.add_argument('--character', type=int, default=0, help='Enter the Character ID you wish to track if all of '
                                                            'your images are all the same character.')
    p.set_defaults(func=fix_manual_alignments)

    def clear_workspace(arrrgs):
        print("Clearing the workspace.")
        workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
        data_dst = os.path.join(workspace, 'data_dst')
        data_src = os.path.join(workspace, 'data_src')
        shutil.rmtree(data_dst)
        shutil.rmtree(data_src)
        print("Done.")

    p = subparsers.add_parser("clear-workspace")
    p.set_defaults(func=clear_workspace)

    def process_extract(arrrgs):
        os_utils.set_process_lowest_prio()
        if arrrgs.manual or arrrgs.character > 0:
            arrrgs.detector = 'manual'
        from mainscripts import Extractor
        Extractor.main(arrrgs.input_dir,
                       arrrgs.output_dir,
                       arrrgs.debug_dir,
                       arrrgs.detector,
                       arrrgs.manual_fix,
                       arrrgs.manual_output_debug_fix,
                       arrrgs.manual_window_size,
                       face_type=arrrgs.face_type,
                       device_args={'cpu_only': arrrgs.cpu_only, 'multi_gpu': arrrgs.multi_gpu},
                       character_number=arrrgs.character,
                       gamma=arrrgs.gamma
                       )

    p = subparsers.add_parser("extract", help="Extract the faces from a pictures.")
    p.add_argument('--input', default='workspace/data_dst', action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output', default=DATA_DST_ALIGNED, action=FixPathAction, dest="output_dir",
                   help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--debug', default=DEBUG_EXTRACTION_DIR, action=FixPathAction, dest="debug_dir",
                   help="Writes debug images to this directory. Default location is: '" + DEBUG_EXTRACTION_DIR + "'")
    p.add_argument('--face-type', dest="face_type",
                   choices=['half_face', 'full_face', 'head', 'full_face_no_align', 'mark_only'], default='full_face',
                   help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")
    p.add_argument('--detector', dest="detector", choices=['dlib', 'mt', 's3fd', 'manual'], default='s3fd',
                   help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, "
                        "perfect for gathering thousands faces for src-set. It is also good for dst-set, "
                        "but can generate false faces in frames where main face not recognized! In this case for "
                        "dst-set use either 'dlib' with '--manual-fix' or '--detector manual'. "
                        "Manual detector suitable only for dst-set.")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=False, help="Enables multi GPU.")
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False,
                   help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False,
                   help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1920,
                   help="Manual fix window size. Default: 1920.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False,
                   help="Extract on CPU. Forces to use MT extractor.")
    p.add_argument('--manual', action='store_true', default=False)
    p.add_argument('--character', type=int, required=True, help="Enter the character id you wish to track. \n0=all")
    p.add_argument('--gamma', type=float, default=1.1,
                   help='Image brightness may be adjusted for better extractions 1.0=no brightness 1.4=high-brightness')
    p.set_defaults(func=process_extract)


    def open_file(path):
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])


    def sort_person(arrrgs):
        os_utils.set_process_lowest_prio()
        from mainscripts import Sorter
        person_uuid = arrrgs.person_uuid
        db = db_connection.open_db_connection()
        query = f"SELECT COPYRIGHT FROM inconode where CONTENT_TYPE='extracted' AND DISTRIBUTION='{person_uuid}'"
        cursor = db.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        yymmdd = ''
        if len(result):
            yymmdd = result[0][0]
        if yymmdd == '':
            print(f"\nError, could not find person: {person_uuid} \nPerhaps they've not been extracted, "
                  f"or have a different CONTENT_TYPE code in the database.")
            exit(-1)

        training_data_src_dir = os.path.join(arrrgs.base, live_faces, yymmdd, person_uuid, 'chips_224')
        Sorter.main(input_path=training_data_src_dir, sort_by_method='hist')
        open_file(training_data_src_dir)

    p = subparsers.add_parser("sort-person", help="Sort the Museum Guest's images")
    p.add_argument('-u', '--person-uuid', dest='person_uuid' )
    p.add_argument('--base', default='/media', help="Default is /media for Linux. Adjust if your filepath base to /warriordata is different (windows or mac)")
    p.set_defaults(func=sort_person)

    def sort_vgg(arrrgs):
        os_utils.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main(input_path=arrrgs.input_dir, sort_by_method=arrrgs.sort_by_method)

    p = subparsers.add_parser("sort-vgg", help='Sort by VGGFace')
    p.add_argument('--input', default=DATA_DST_ALIGNED, action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', default='vggface', dest="sort_by_method",
                   choices=("blur", "face", "face-dissim", "face-yaw", "face-pitch", "hist", "hist-dissim",
                            "brightness", "hue", "black", "origname", "oneface", "final", "final-no-blur",
                            "vggface", "test"),
                   help="Method of sorting. 'origname'=sort by original filename to recover original sequence. "
                        "'vggface' is default")
    p.set_defaults(func=sort_vgg)

    p = subparsers.add_parser("sort-hist", help='Sort by Histogram')
    p.add_argument('--input', default=DATA_DST_ALIGNED, action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process. "
                        "Default is " + DATA_DST_ALIGNED)
    p.add_argument('--by', default='hist', dest="sort_by_method",
                   choices=("blur", "face", "face-dissim", "face-yaw", "face-pitch", "hist", "hist-dissim",
                            "brightness", "hue", "black", "origname", "oneface", "final", "final-no-blur",
                            "vggface", "test"),
                   help="Method of sorting. 'origname' sort by original filename to recover original sequence. "
                        "'hist' Histogram is default.")
    p.set_defaults(func=sort_vgg)


    def recover_original_filenames(arrrgs):
        os_utils.set_process_lowest_prio()
        from mainscripts import Util
        Util.recover_original_aligned_filename(input_path=arguments.input_dir)

    p = subparsers.add_parser("recover-filenames", help="recover the original file names")
    p.add_argument('--input-dir', default=DATA_DST_ALIGNED, action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.set_defaults(func=recover_original_filenames)

    def id_character(arrrgs):
        from utils import assign_character
        assign_character.process_character(arrrgs.input, arrrgs.character)

    p = subparsers.add_parser("assign", help="Assign a character ID to images")
    p.add_argument('-i', '--input', type=str, default=DATA_DST_ALIGNED, help="Directory of images you wish to assign. "
                                                                             "Defaults to workspace/data_dst/aligned")
    p.add_argument('-c', '--character', type=int, required=True, help="The Character ID number (required)")
    p.set_defaults(func=id_character)


    def bad_args(arrrgs):
        parser.print_help()
        exit(0)

    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)
