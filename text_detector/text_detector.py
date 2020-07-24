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

import sys
import os

import logging as log
import cv2
import numpy as np

from openvino.inference_engine import IECore
SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28

class TextDetector():
    
    def __init__(self, args):
        self.args = args
        log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

        # Plugin initialization for specified device and load extensions library if specified.
        log.info('Creating Inference Engine...')
        self.ie = IECore()
        if args.cpu_extension and 'CPU' in args.device:
            ie.add_extension(args.cpu_extension, 'CPU')
        # Read IR
        log.info('Loading Mask-RCNN network')
        mask_rcnn_net = self.ie.read_network(args.mask_rcnn_model, os.path.splitext(args.mask_rcnn_model)[0] + '.bin')

        log.info('Loading encoder part of text recognition network')
        text_enc_net = self.ie.read_network(args.text_enc_model, os.path.splitext(args.text_enc_model)[0] + '.bin')

        log.info('Loading decoder part of text recognition network')
        text_dec_net = self.ie.read_network(args.text_dec_model, os.path.splitext(args.text_dec_model)[0] + '.bin')

        if 'CPU' in args.device:
            supported_layers = self.ie.query_network(mask_rcnn_net, 'CPU')
            not_supported_layers = [l for l in mask_rcnn_net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error('Following layers are not supported by the plugin for specified device {}:\n {}'.
                          format(args.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        required_input_keys = {'im_data', 'im_info'}
        assert required_input_keys == set(mask_rcnn_net.inputs.keys()), \
            'Demo supports only topologies with the following input keys: {}'.format(', '.join(required_input_keys))
        required_output_keys = {'boxes', 'scores', 'classes', 'raw_masks', 'text_features'}
        assert required_output_keys.issubset(mask_rcnn_net.outputs.keys()), \
            'Demo supports only topologies with the following output keys: {}'.format(', '.join(required_output_keys))

        self.n, self.c, self.h, self.w = mask_rcnn_net.inputs['im_data'].shape
        assert self.n == 1, 'Only batch 1 is supported by the demo application'

        log.info('Loading IR to the plugin...')
        self.mask_rcnn_exec_net = self.ie.load_network(network=mask_rcnn_net, device_name=args.device, num_requests=2)
        self.text_enc_exec_net = self.ie.load_network(network=text_enc_net, device_name=args.device)
        self.text_dec_exec_net = self.ie.load_network(network=text_dec_net, device_name=args.device)

        self.hidden_shape = text_dec_net.inputs[args.trd_input_prev_hidden].shape
        
    def expand_box(self, box, scale):
        w_half = (box[2] - box[0]) * .5
        h_half = (box[3] - box[1]) * .5
        x_c = (box[2] + box[0]) * .5
        y_c = (box[3] + box[1]) * .5
        w_half *= scale
        h_half *= scale
        box_exp = np.zeros(box.shape)
        box_exp[0] = x_c - w_half
        box_exp[2] = x_c + w_half
        box_exp[1] = y_c - h_half
        box_exp[3] = y_c + h_half
        return box_exp
        
          
    def segm_postprocess(self, box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
        extended_box = self.expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                                (x0 - extended_box[0]):(x1 - extended_box[0])]
        return im_mask


    def sameLine(self,box,boxes):
        margin=[]
        c=(box[3]-box[1])/4
        for i in len(boxes):
            if box[1]-margin >= boxes[i][0]-c & box[1] <= boxes[i][0]+margin:
                margin.append(i)

        return margin



    def process(self, frame):
        if not self.args.keep_aspect_ratio:
            # Resize the image to a target size.
            scale_x = self.w / frame.shape[1]
            scale_y = self.h / frame.shape[0]
            input_image = cv2.resize(frame, (self.w, self.h))
        else:
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(self.h / frame.shape[0], self.w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, self.h - input_image_size[0]),
                                           (0, self.w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)

        # Run the net.
        outputs = self.mask_rcnn_exec_net.infer({'im_data': input_image, 'im_info': input_image_info})

        # Parse detection results of the current request
        boxes = outputs['boxes']
        scores = outputs['scores']
        classes = outputs['classes'].astype(np.uint32)
        raw_masks = outputs['raw_masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > self.args.prob_threshold
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]


        


        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            raw_cls_mask = raw_mask[cls, ...]
            mask = self.segm_postprocess(box, raw_cls_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        texts = []
        for feature in text_features:
            feature = self.text_enc_exec_net.infer({'input': feature})['output']
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(self.hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for i in range(MAX_SEQ_LEN):
                decoder_output = self.text_dec_exec_net.infer({
                    self.args.trd_input_prev_symbol: prev_symbol_index,
                    self.args.trd_input_prev_hidden: hidden,
                    self.args.trd_input_encoder_outputs: feature})
                symbols_distr = decoder_output[self.args.trd_output_symbols_distr]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += self.args.alphabet[prev_symbol_index]
                hidden = decoder_output[self.args.trd_output_cur_hidden]

            texts.append(text)
        return (boxes, classes, scores, masks, texts)





