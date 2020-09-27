#!/usr/bin/env python
"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function

import logging as log
import os
import sys
import time
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np
import re
import datetime

import json

from document_identifier import DocumentIdentifier
from text_detector import TextDetector 
from web_server import WebServer 
from utils.tracker import StaticIOUTracker



class FolderCapture:
    def __init__(self, path):
        self.images_paths = []
        self.current_index = 0
        for imname in os.listdir(path):
            if imname.lower().endswith('.jpg') or imname.lower().endswith('.png'):
                self.images_paths.append(os.path.join(path, imname))

    def read(self):
        ret = False
        image = None
        if self.current_index < len(self.images_paths):
            image = cv2.imread(self.images_paths[self.current_index])
            ret = True
            self.current_index += 1

        return ret, image

    def isOpened(self):
        return len(self.images_paths) > 0

    def release(self):
        self.images_paths = []

def build_argparser():
        parser = ArgumentParser(add_help=False)
        args = parser.add_argument_group('Options')
        args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                          help='Show this help message and exit.')
        args.add_argument('-m_m', '--mask_rcnn_model',
                          help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                               'additional text features output.',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-m_te', '--text_enc_model',
                          help='Required. Path to an .xml file with a trained text recognition model '
                               '(encoder part).',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-m_td', '--text_dec_model',
                          help='Required. Path to an .xml file with a trained text recognition model '
                                 '(decoder part).',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-m_fd', '--face_detect_model',
                          help='Required. Path to an .xml file with a trained Face-Detection model with ',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-m_lm', '--land_marks_model',
                          help='Required. Path to an .xml file with a trained Facial-Landmarks model with ',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-i',
                          dest='input_source',
                          help='Required. Path to an image, video file or a numeric camera ID.',
                          required=True, type=str, metavar='"<path>"')
        args.add_argument('-d', '--device',
                          help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                               'The demo will look for a suitable plugin for device specified '
                               '(by default, it is CPU).',
                          default='CPU', type=str, metavar='"<device>"')
        args.add_argument('-l', '--cpu_extension',
                          help='Required for CPU custom layers. '
                               'Absolute path to a shared library with the kernels implementation.',
                          default=None, type=str, metavar='"<absolute_path>"')
        args.add_argument('--delay',
                          help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                          default=0, type=int, metavar='"<num>"')
        args.add_argument('-pt', '--prob_threshold',
                          help='Optional. Probability threshold for detections filtering.',
                          default=0.5, type=float, metavar='"<num>"')
        args.add_argument('-a', '--alphabet',
                          help='Optional. Alphabet that is used for decoding.',
                          default='  0123456789abcdefghijklmnopqrstuvwxyz')
        args.add_argument('--trd_input_prev_symbol',
                          help='Optional. Name of previous symbol input node to text recognition head decoder part.',
                          default='prev_symbol')
        args.add_argument('--trd_input_prev_hidden',
                          help='Optional. Name of previous hidden input node to text recognition head decoder part.',
                          default='prev_hidden')
        args.add_argument('--trd_input_encoder_outputs',
                          help='Optional. Name of encoder outputs input node to text recognition head decoder part.',
                          default='encoder_outputs')
        args.add_argument('--trd_output_symbols_distr',
                          help='Optional. Name of symbols distribution output node from text recognition head decoder part.',
                          default='output')
        args.add_argument('--trd_output_cur_hidden',
                          help='Optional. Name of current hidden output node from text recognition head decoder part.',
                          default='hidden')
        args.add_argument('--keep_aspect_ratio',
                          help='Optional. Force image resize to keep aspect ratio.',
                          action='store_true')
        args.add_argument('--no_track',
                          help='Optional. Disable tracking.',
                          action='store_true')
        args.add_argument('--show_scores',
                          help='Optional. Show detection scores.',
                          action='store_true')
        args.add_argument('--show_boxes',
                          help='Optional. Show bounding boxes.',
                          action='store_true')
        args.add_argument('-c', '--config',
                          help='Required. Path to an .xml file with a trained Mask-RCNN model with '
                               'additional text features output.',
                          required=True, type=str, metavar='"<path>"')
        return parser
       

def main():

        log.basicConfig(format='[ %(levelname)s ] %(message)s' , level=log.INFO, stream=sys.stdout)
        args = build_argparser().parse_args()
        text_detector = TextDetector(args) 
        document_identifier = DocumentIdentifier(args.config)
        web_server = WebServer()
        web_server.start()
        

        try:
            input_source = int(args.input_source)
        except ValueError:
            input_source = args.input_source

        if os.path.isdir(input_source):
            cap = FolderCapture(input_source)
        else:
            cap = cv2.VideoCapture(input_source)

        if not cap.isOpened():
            log.error('Failed to open "{}"'.format(args.input_source))
        if isinstance(cap, cv2.VideoCapture):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        


        log.info('Starting inference...')
        print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
        
            text = text_detector.process(frame, True)

            result=document_identifier.process(text)

            print(json.dumps(result, indent=4))

            while True:
              #web_server.push(result)
              cv2.waitKey(4000)

         

            key = cv2.waitKey(0)
            esc_code = 27
            if key == esc_code:
              break
            cv2.destroyAllWindows()
            cap.release()


if __name__ == '__main__':
	sys.exit(main() or 0)
