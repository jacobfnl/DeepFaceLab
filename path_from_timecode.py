""" Usage:
    from path_from_timecode import WarriorsSourceImages

    warriors_src = WarriorsSourceImages()
    # Now, request a path for an image with either timecode or frame number
"""
import os
from timecode import Timecode


class WarriorsSourceImages:
    def __init__(self, base_path='/media/warriordata'):
        if base_path[-1] != '/':
            base_path += '/'
        base_path += 'WARRIORS_SRC/F'
        # Test for access to frame 1
        test_path = base_path + '11/warriors_src.114735.png'
        # if not os.path.isfile(test_path):
        #     print("Error! Path to WARRIORS_SRC is inaccessible. \nCould not find file a path:\n{}".format(test_path))
        #     exit(-1)
        self.base_path = base_path
        self.tc24 = Timecode('24')


    def path_for_frame(self, frame_number: int):
        if frame_number < 1 or frame_number > 135216:  # known frame range
            print('ERROR: frame out of range')
            exit(-1)
        frame = frame_number // 10000
        return self.base_path + str(frame).zfill(2) + '/warriors_src.' + str(frame_number).zfill(6) + '.png'

    def frame_for_timecode(self, timecode: str):
        self.tc24.set_timecode(timecode)
        return self.tc24.frame_number + 1  # solves for off by 1 error in timecode

    def path_for_timecode(self, timecode: str):
        # return a frame path for timecode
        frame_number = self.frame_for_timecode(timecode)
        return self.path_for_frame(frame_number)

    def timecode_for_frame(self, frame_number: str):
        h,m,s,f = self.tc24.frames_to_tc(int(frame_number))
        return str(h).zfill(2)+':'+str(m).zfill(2)+':'+str(s).zfill(2)+':'+str(f).zfill(2)


if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser(description="To grab a frame file path from WARRIORS_SRC.\
        Use the --timecode option in the format of 01:11:11:11 OR the --frame option with a frame number\
        to receive file path for timecode/frame.")

    arg_parser.add_argument('--base', type=str, default='/media/warriordata', help='location of WARRIORS_SRC')
    arg_parser.add_argument('-f', '--frame', type=str, default='',
                            help='the frame number for which you would like a file path')
    arg_parser.add_argument('-t', '--timecode', type=str, default='',
                            help='the timecode for which you would like a frame path')
    arg_parser.add_argument('--frame-number', action="store_true", dest="fn_only", default=False, help="Frame number only")
    arg_parser.add_argument('--generate-time', action="store_true", dest="generate_time", default=False, help="Timecode for framenumber")

    args = arg_parser.parse_args()
    if args.frame is '' and args.timecode is '':
        print(arg_parser.print_help())
        exit()

    warriors_src = WarriorsSourceImages(args.base)

    if args.timecode is not '':
        if args.fn_only:
            print(warriors_src.frame_for_timecode(args.timecode))
        else:
            print(warriors_src.path_for_timecode(args.timecode))
    elif args.generate_time:
        print(warriors_src.timecode_for_frame(args.frame))

    else:  # if run independently, we need to handle frame args
        print(warriors_src.path_for_frame(int(args.frame)))
