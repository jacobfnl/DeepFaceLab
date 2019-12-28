﻿import cv2
from pathlib import Path
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from facelib import LandmarksProcessor
from interact import interact as io

def remove_ie_polys_file (filepath):
    filepath = Path(filepath)

    if filepath.suffix == '.png':
        dflimg = DFLPNG.load( str(filepath) )
    elif filepath.suffix == '.jpg':
        dflimg = DFLJPG.load ( str(filepath) )
    else:
        return

    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dflimg.remove_ie_polys()
    dflimg.embed_and_set( str(filepath) )


def remove_ie_polys_folder(input_path):
    input_path = Path(input_path)

    io.log_info ("Removing ie_polys...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Removing"):
        filepath = Path(filepath)
        remove_ie_polys_file(filepath)
        
def remove_fanseg_file (filepath):
    filepath = Path(filepath)

    if filepath.suffix == '.png':
        dflimg = DFLPNG.load( str(filepath) )
    elif filepath.suffix == '.jpg':
        dflimg = DFLJPG.load ( str(filepath) )
    else:
        return

    if dflimg is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dflimg.remove_fanseg_mask()
    dflimg.embed_and_set( str(filepath) )


def remove_fanseg_folder(input_path):
    input_path = Path(input_path)

    io.log_info ("Removing fanseg mask...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Removing"):
        filepath = Path(filepath)
        remove_fanseg_file(filepath)

def convert_png_to_jpg_file (filepath, output_path=None):
    filepath = Path(filepath)

    if filepath.suffix != '.png':
        return

    dflpng = DFLPNG.load (str(filepath) )
    if dflpng is None:
        io.log_err ("%s is not a dfl image file" % (filepath.name) )
        return

    dfl_dict = dflpng.getDFLDictData()

    img = cv2_imread (str(filepath))
    orig_output_path = output_path
    if output_path is None:
        output_path = filepath.parent
    else:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        if not output_path.is_dir():
            output_path.mkdir(parents=True)
    new_filepath = str(output_path / (filepath.stem + '.jpg'))
    cv2_imwrite ( new_filepath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    DFLJPG.embed_data( new_filepath,
                       face_type=dfl_dict.get('face_type', None),
                       landmarks=dfl_dict.get('landmarks', None),
                       ie_polys=dfl_dict.get('ie_polys', None),
                       source_filename=dfl_dict.get('source_filename', None),
                       source_rect=dfl_dict.get('source_rect', None),
                       source_landmarks=dfl_dict.get('source_landmarks', None) )
    if orig_output_path is None:
        filepath.unlink()

def convert_png_to_jpg_folder (input_path, output_path=None):
    input_path = Path(input_path)

    io.log_info ("Converting PNG to JPG...\r\n")

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Converting"):
        filepath = Path(filepath)
        convert_png_to_jpg_file(filepath, output_path=output_path)

def add_landmarks_debug_images(input_path, output_path=None):
    io.log_info ("Adding landmarks debug images...")

    if output_path is not None:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        if not output_path.is_dir():
            output_path.mkdir(parents=True)

    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        img = cv2_imread(str(filepath))

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        if img is not None:
            face_landmarks = dflimg.get_landmarks()
            LandmarksProcessor.draw_landmarks(img, face_landmarks, transparent_mask=True, ie_polys=dflimg.get_ie_polys() )

            if output_path is None:
                output_file = '{}{}'.format( str(Path(str(input_path)) / filepath.stem),  '_debug.jpg')
            else:
                output_file = '{}{}'.format(output_path / filepath.stem, '.jpg')
            cv2_imwrite(output_file, img, [int(cv2.IMWRITE_JPEG_QUALITY), 50] )

def recover_original_aligned_filename(input_path):
    io.log_info ("Recovering original aligned filename...")

    files = []
    for filepath in io.progress_bar_generator( Path_utils.get_image_paths(input_path), "Processing"):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        files += [ [filepath, None, dflimg.get_source_filename(), False, dflimg.get_character_number()] ]

    files_len = len(files)
    for i in io.progress_bar_generator( range(files_len), "Sorting" ):
        fp, _, sf, converted, cn = files[i]

        if converted:
            continue

        sf_stem = Path(sf).stem
        generic = ''
        if cn is None:
            generic = '_0'
        files[i][1] = fp.parent / ( sf_stem + generic + fp.suffix )
        files[i][3] = True
        c = 1

        for j in range(i+1, files_len):
            fp_j, _, sf_j, converted_j, cn_j = files[j]
            if converted_j:
                continue

            if sf_j == sf:
                generic = ''
                if cn_j is None:
                    generic = "_{}".format(c)
                files[j][1] = fp_j.parent / ( sf_stem + generic + fp_j.suffix )
                files[j][3] = True
                c += 1

    for file in io.progress_bar_generator( files, "Renaming", leave=False ):
        fs, _, _, _, _ = file
        dst = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (dst)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )

    for file in io.progress_bar_generator( files, "Renaming" ):
        fs, fd, _, _, _ = file
        fs = fs.parent / ( fs.stem + '_tmp' + fs.suffix )
        try:
            fs.rename (fd)
        except:
            io.log_err ('fail to rename %s' % (fs.name) )
