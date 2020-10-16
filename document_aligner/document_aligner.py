import os
import sys
import logging as log
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class Network:
    def __init__(self, ie, model_xml, device, num_requests=0):
        self.ie = ie
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)

        assert len(self.net.inputs) == 1, "Expected 1 input blob"
        assert len(self.net.outputs) == 1, "Expected 1 output blob"
        
        self.net.batch_size = 1

        for key in self.net.inputs.keys():
                # Should be called before load of the network to the plugin
            self.net.inputs[key].precision = 'U8'

        self.exec_net = self.ie.load_network(self.net, device, num_requests=num_requests)
        
        self.input_blob = next(iter(self.net.inputs))
        self.n, self.c, self.height, self.width = self.net.inputs[self.input_blob].shape

        self.out_blob = next(iter(self.net.outputs))

class DocumentAligner():

    def __init__(self, args):
        self.ie = IECore()
        self.face_detect_net = Network(self.ie, args.face_detect_model, "CPU", 1)
        self.landmarks_net = Network(self.ie, args.land_marks_model, "CPU", 1)
        
       
    def preprocess(self, frame):
        in_frame = frame.copy()
        fh, fw = in_frame.shape[:-1]
        if (self.face_detect_net.height, self.face_detect_net.width) != (fh, fw):
            in_frame = cv2.resize(in_frame, (self.face_detect_net.width, self.face_detect_net.height))
        in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        res = self.face_detect_net.exec_net.infer({self.face_detect_net.input_blob: in_frame})
        res = res[self.face_detect_net.out_blob]
        data = res[0][0]
        boxes = []
        for number, proposal in enumerate(data):
            image_id = np.int(proposal[0])
            if image_id >= 0:
                confidence = proposal[2]
                if confidence >= 0.5:
                    xmin = np.int(fw * proposal[3])
                    ymin = np.int(fh * proposal[4])
                    xmax = np.int(fw * proposal[5])
                    ymax = np.int(fh * proposal[6])
                    boxes.append([xmin, ymin, xmax, ymax])   
        

        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
        return frame
        

		