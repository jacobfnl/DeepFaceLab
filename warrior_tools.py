import os
import shutil
import multiprocessing

from mainscripts import Extractor
from mainscripts.Util import recover_original_aligned_filename
from utils import Path_utils, os_utils
from interact import interact as io
from pathlib import Path
from shutil import copyfile
import argparse

DATA_DST_ALIGNED = 'workspace/data_dst/aligned'


class FixPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def fix_manual_cache(input_path, source_path):
    recover_original_aligned_filename(input_path)
    files = []
    source_chosen = os.path.join(source_path, 'chosen_frames')
    if not os.path.exists(source_chosen):
        os.makedirs(source_chosen)

    for filepath in io.progress_bar_generator(Path_utils.get_image_paths(input_path),
                                              "Processing New Manual Fix Order."):
        filepath = Path(filepath)
        # find the source image in source path.
        # example /home/cyrus/Documents/DFL/workspace/data_dst/warriors_src.086299_0.jpg
        file_parts = filepath.stem.split('.')

        chosen_file = filepath.stem.split('_')[0] + '.png'

        if os.path.exists(os.path.join(source_path, chosen_file)):
            print("found png: {}".format(chosen_file))
            exit()
        else:
            source_jpg = chosen_file.split('.')[0] + '.jpg'  # ¯\_(ツ)_/¯
            s_png = chosen_file
            if os.path.exists(os.path.join(source_path, source_jpg)):
                print("found jpg: {}".format(source_jpg))
                chosen_file = source_jpg
                exit()
            else:
                print("Unable to find path: {}\nand not here either: {}".format(os.path.join(source_path, s_png),
                                                                                os.path.join(source_path, source_jpg)))
                exit()
        # copy chosen originals to source_chosen

        copyfile(os.path.join(source_path, chosen_file), os.path.join(source_chosen, chosen_file))

    # now implement manual fix on the items.
    '''
    "%PYTHON_EXECUTABLE%" "%DFL_ROOT%\main.py"
    --input-dir  "%WORKSPACE%\data_dst"
    --output-dir  "%WORKSPACE%\data_dst\aligned"
    --multi-gpu
    --detector manual --debug-dir "%WORKSPACE%\data_dst\aligned_debug"
    '''

    Extractor.main(source_chosen,
                   input_path,
                   None,
                   'manual',
                   False,
                   False,
                   1920,
                   face_type='full_face',
                   device_args={'multi_gpu': True}
                   )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()


    def fix_manual_alignments(arrrgs):
        fix_manual_cache(arrrgs.input, arrrgs.source)


    p = subparsers.add_parser("fix-manual-cache")
    p.add_argument('-i', '--input', type=str, required=True, help='Directory of aligned images you wish to re-do.')
    p.add_argument('-s', '--source', type=str, required=True, help='Directory of Source Frames.')
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
        from mainscripts import Extractor
        Extractor.main(arrrgs.input_dir,
                       arrrgs.output_dir,
                       arrrgs.debug_dir,
                       arrrgs.detector,
                       arrrgs.manual_fix,
                       arrrgs.manual_output_debug_fix,
                       arrrgs.manual_window_size,
                       face_type=arrrgs.face_type,
                       device_args={'cpu_only': arrrgs.cpu_only,
                                    'multi_gpu': arrrgs.multi_gpu}
                       )


    p = subparsers.add_parser("extract", help="Extract the faces from a pictures.")
    p.add_argument('--input-dir', default='workspace/data_dst', action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--output-dir', default='workspace/data_dst/aligned', action=FixPathAction, dest="output_dir",
                   help="Output directory. This is where the extracted files will be stored.")
    p.add_argument('--debug-dir', default='workspace/data_dst/debug_extraction', action=FixPathAction, dest="debug_dir",
                   help="Writes debug images to this directory.")
    p.add_argument('--face-type', dest="face_type",
                   choices=['half_face', 'full_face', 'head', 'full_face_no_align', 'mark_only'], default='full_face',
                   help="Default 'full_face'. Don't change this option, currently all models uses 'full_face'")
    p.add_argument('--detector', dest="detector", choices=['dlib', 'mt', 's3fd', 'manual'], default='s3fd',
                   help="Type of detector. Default 'dlib'. 'mt' (MTCNNv1) - faster, better, almost no jitter, perfect for gathering thousands faces for src-set. It is also good for dst-set, but can generate false faces in frames where main face not recognized! In this case for dst-set use either 'dlib' with '--manual-fix' or '--detector manual'. Manual detector suitable only for dst-set.")
    p.add_argument('--multi-gpu', action="store_true", dest="multi_gpu", default=True, help="Enables multi GPU.")
    p.add_argument('--manual-fix', action="store_true", dest="manual_fix", default=False,
                   help="Enables manual extract only frames where faces were not recognized.")
    p.add_argument('--manual-output-debug-fix', action="store_true", dest="manual_output_debug_fix", default=False,
                   help="Performs manual reextract input-dir frames which were deleted from [output_dir]_debug\ dir.")
    p.add_argument('--manual-window-size', type=int, dest="manual_window_size", default=1920,
                   help="Manual fix window size. Default: 1920.")
    p.add_argument('--cpu-only', action="store_true", dest="cpu_only", default=False,
                   help="Extract on CPU. Forces to use MT extractor.")
    p.set_defaults(func=process_extract)


    def sort_vgg(arrrgs):
        os_utils.set_process_lowest_prio()
        from mainscripts import Sorter
        Sorter.main(input_path=arrrgs.input_dir, sort_by_method=arrrgs.sort_by_method)


    p = subparsers.add_parser("sort-vgg")
    p.add_argument('--input-dir', default=DATA_DST_ALIGNED, action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.add_argument('--by', required=True, dest="sort_by_method", choices=(
    "blur", "face", "face-dissim", "face-yaw", "face-pitch", "hist", "hist-dissim", "brightness", "hue", "black",
    "origname", "oneface", "final", "final-no-blur", "vggface", "test"),
                   help="Method of sorting. 'origname' sort by original filename to recover original sequence.")
    p.set_defaults(func=sort_vgg)


    def recover_original_filenames(arrrgs):
        os_utils.set_process_lowest_prio()
        from mainscripts import Util
        if arguments.recover_original_aligned_filename:
            Util.recover_original_aligned_filename(input_path=arguments.input_dir)


    p = subparsers.add_parser("recover-filenames")
    p.add_argument('--input-dir', default=DATA_DST_ALIGNED, action=FixPathAction, dest="input_dir",
                   help="Input directory. A directory containing the files you wish to process.")
    p.set_defaults(func=recover_original_filenames)


    def bad_args(arrrgs):
        parser.print_help()
        exit(0)


    parser.set_defaults(func=bad_args)
    arguments = parser.parse_args()
    arguments.func(arguments)


