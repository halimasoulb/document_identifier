import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore



class DocumentAligner():

    def __init__(self, args):

        self.args=args
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

        assert len(face_detect_net.inputs) == 1, "Expected 1 input blob"
        assert len(face_detect_net.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(face_detect_net.inputs))
        self.output_blob = next(iter(face_detect_net.outputs))

        log.info('Loading IR to the plugin...')
        self.face_detect_exec_net = self.ie.load_network(network=face_detect_net, device_name=args.device, num_requests=2)
        self.land_marks_exec_net = self.ie.load_network(network=land_marks_net, device_name=args.device)

        self.input_shape = face_detect_net.inputs[self.input_blob].shape
        self.output_shape = face_detect_net.outputs[self.output_blob].shape


    def process(self, frame):
        return frame
        
    
       
              
        
    	
        

		