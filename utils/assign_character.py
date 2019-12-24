import os
import shutil
from utils import Path_utils
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG
from pathlib import Path
from interact import interact as io
from mainscripts.Util import recover_original_aligned_filename


def process_character(input_dir, character_id: int):
    # recover_original_aligned_filename(input_dir)
    input_path = Path(input_dir)
    files = []
    for filepath in io.progress_bar_generator(Path_utils.get_image_paths(input_path),
                                              "Processing Character id: " + str(character_id)):
        filepath = Path(filepath)
        dflimg = None
        if filepath.suffix == '.png':
            dflimg = DFLPNG.load(str(filepath))
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load(str(filepath))
        if dflimg is None:
            io.log_err("{} is not a dfl image file".format(filepath.name))
            continue
        source = dflimg.get_source_filename()
        split_stem = source.split('_')
        new_file = split_stem[0] + '_' + split_stem[1] + '_' + str(character_id) + filepath.suffix
        if source == new_file:
            print("character id is already set.")
            continue
        dir = filepath.parent
        try:
            filepath.rename(os.path.join(dir, new_file))
        except:
            io.log_err("failed to rename {}".format(filepath.name))

        dflimg.embed_and_set(os.path.join(dir, new_file), source_filename=new_file)

