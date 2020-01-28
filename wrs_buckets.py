import os
import shutil
from pathlib import Path
from path_from_timecode import WarriorsSourceImages
from utils.db_connection import open_db_connection
from interact import interact as io


server_root = "/media/warriordata"

def get_film_characters():
    cursor = db.cursor()
    query = "SELECT ID, LINK_IDS FROM filmcharacter"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


def get_dfl_chips(link: str):
    cursor = db.cursor()
    query = f"SELECT ID, CONTEXT, URL FROM DFLconode WHERE LINK_IDS='{link}'"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result


def make_link_paths(link:str):
    link_path = os.path.join(server_root, "character_buckets", link_id)
    chips_path = os.path.join(link_path, "chips")
    if not os.path.exists(chips_path):
        os.makedirs(chips_path)
    frames_path = os.path.join(link_path, "frames")
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    output_path = os.path.join(link_path, "output", "_latest")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return link_path, chips_path, frames_path, output_path


if __name__ == "__main__":
    warriors_source_images = WarriorsSourceImages(server_root)
    db = open_db_connection()
    film_characters = get_film_characters()
    x = 0
    errors = []
    clear_alpha_path = os.path.join(server_root, "testing", "clear_alpha.png")
    for row in film_characters:
        x += 1
        film_kid, link_id = row
        print(f"character: {link_id}")
        link_path, chips_path, frames_path, output_path = make_link_paths(link_id)
        # grab chips info
        dfl_conode = get_dfl_chips(link_id)
        io.progress_bar("Collecting files", len(dfl_conode))
        for dfl_row in dfl_conode:
            dfl_id, frame_str, chip_url = dfl_row
            frame_str: str = frame_str.split("_")[0]
            frame_number = int(frame_str)
            chip_url = server_root + chip_url[4:]
            try:
                frame_url = warriors_source_images.path_for_frame(frame_number)
            except ValueError:
                errors.append(f"err: dfl_row: {dfl_id}, frame: {frame_str}, could not get frame url from frame number.")
                io.progress_bar_inc(1)
                continue
            # print(f"chip_URL: {chip_url}\tframe:{frame_url}")
            c_path = Path(chip_url)
            c_dest = os.path.join(chips_path, c_path.stem + c_path.suffix)
            f_path = Path(frame_url)
            f_dest = os.path.join(frames_path, f_path.stem + f_path.suffix)
            out_dest = os.path.join(output_path, f_path.stem + f_path.suffix)
            if not os.path.exists(c_dest):
                shutil.copyfile(chip_url, c_dest)
            if not os.path.exists(f_dest):
                shutil.copyfile(frame_url, f_dest)
            if not os.path.exists(out_dest):
                shutil.copyfile(clear_alpha_path, out_dest)
            io.progress_bar_inc(1)
        io.progress_bar_close()

    for err in errors:
        print(err)