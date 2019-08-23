import multiprocessing
import os
import unittest
from utils import os_utils

INPUT_DIR = os.path.abspath("test/assets/single-face-01")
# INPUT_DIR = os.path.join(ASSETS, 'face')
OUTPUT_DIR = os.path.join(INPUT_DIR, 'aligned')


class SmokeTest(unittest.TestCase):
    def test_extraction(self):
        multiprocessing.set_start_method("spawn")
        os_utils.set_process_lowest_prio()
        from mainscripts import Extractor

        Extractor.main(INPUT_DIR, OUTPUT_DIR, device_args={'cpu_only': True})
        output_dir_items = list(os.scandir(OUTPUT_DIR))

        # assert that the aligned directory contains a single extracted face
        self.assertEqual(len(output_dir_items), 1)
        extracted_file = output_dir_items[0]
        self.assertTrue(extracted_file.is_file())
        self.assertEqual(extracted_file.name, 'latent0243_0.png')


if __name__ == '__main__':
    unittest.main()
