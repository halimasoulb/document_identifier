import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore



class DocumentAligner():

    def __init__(self, args):

        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.args = args

        log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

        # Plugin initialization for specified device and load extensions library if specified.
        log.info('Creating Inference Engine...')
        self.ie = IECore()
        if args.cpu_extension and 'CPU' in args.device:
            ie.add_extension(args.cpu_extension, 'CPU')
        # Read IR
        log.info('Loading face detection network')
        face_detect_net = self.ie.read_network(args.face_detect_model, os.path.splitext(args.face_detect_model)[0] + '.bin')

        log.info('Loading facial landmarks network')
        land_marks_net = self.ie.read_network(args.land_marks_model, os.path.splitext(args.land_marks_model)[0] + '.bin')

        
        if 'CPU' in args.device:
            supported_layers = self.ie.query_network(face_detect_net, 'CPU')
            not_supported_layers = [l for l in face_detect_net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error('Following layers are not supported by the plugin for specified device {}:\n {}'.
                          format(args.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)


        # Face Detection Model
        #  Define Input&Output Network dict keys
        FACEDETECT_INPUTKEYS = 'data'
        FACEDETECT_OUTPUTKEYS = 'detection_out'
        #  Obtain image_count, channels, height and width
        self.n, self.c, self.h, self.w = face_detect_net.inputs[FACEDETECT_INPUTKEYS].shape

        log.info("Loading IR to the plugin...")
        self.face_detect_net_exec = self.ie.load_network(network=face_detect_net, device_name=args.device, num_requests=2)
        self.land_marks_net_exec = self.ie.load_network(network=land_marks_net, device_name=args.device)
        

        self.input_blob = next(iter(face_detect_net.inputs))
        self.output_blob = next(iter(face_detect_net.outputs))

        self.input_shape = face_detect_net.inputs[self.input_blob].shape
        self.output_shape = face_detect_net.outputs[self.output_blob].shape


    def preprocess(self, frame):
        input_blob = cv2.resize(frame, (self.w, self.h))  # Resize width & height
        input_blob = input_blob.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        input_blob = input_blob.reshape((self.n, self.c, self.h, self.w))
        return input_blob
        

        

        
    	
        

		