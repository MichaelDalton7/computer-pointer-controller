import cv2
import numpy as np
import logging as log
import time
import math
from openvino.inference_engine import IECore

'''
Class for the Gaze Estimation Model.
'''
class GazeEstimation:
   
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Use this to set your instance variables.
        '''
        self.core = IECore()
        self.model_name = model_name
        self.extensions = extensions
        self.model_structure = self.model_name + '.xml'
        self.model_weights = self.model_name + '.bin'
        self.device = device
        self.network = None
        self.model = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None

    def load_model(self):
        '''
        You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        # Load extensions if there are any
        if not self.extensions == None:
            self.core.add_extension(self.extensions, self.device)
        # Load model
        log.info("Loading Gaze Estimation model...")
        start_time = time.time()
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        finish_time = time.time()
        log.info("Gaze Estimation model took {} seconds to load.".format(finish_time - start_time))
        # Get Input shape
        self.input_name = [i for i in self.model.inputs.keys()]
        self.input_shape = self.model.inputs[self.input_name[1]].shape
        # Get Output name
        self.output_name = [i for i in self.model.outputs.keys()]

    def predict(self,  left_eye_image, right_eye_image, head_pose_angle):
        '''
        You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_eye_image, processed_right_eye_image = self.preprocess_input(left_eye_image.copy(), right_eye_image.copy())
        outputs = self.network.infer({
            'head_pose_angles' : head_pose_angle, 
            'left_eye_image' : processed_left_eye_image, 
            'right_eye_image' : processed_right_eye_image
        })
        return self.preprocess_output(outputs, head_pose_angle)

    def preprocess_input(self, left_eye_image, right_eye_image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_left_eye_image = cv2.resize(left_eye_image, (self.input_shape[3], self.input_shape[2]))
        processed_left_eye_image = np.transpose(np.expand_dims(processed_left_eye_image, axis=0), (0, 3, 1, 2))
        processed_right_eye_image = cv2.resize(right_eye_image,(self.input_shape[3], self.input_shape[2]))
        processed_right_eye_image = np.transpose(np.expand_dims(processed_right_eye_image, axis=0), (0, 3, 1, 2))
        return processed_left_eye_image, processed_right_eye_image

    def preprocess_output(self, outputs, head_pose_angle):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        angle_r_fc = head_pose_angle[2]
        cosine = math.cos(angle_r_fc * math.pi / 180.0)
        sine = math.sin(angle_r_fc * math.pi / 180.0)
        gaze_vector = outputs[self.output_name[0]][0]
        
        x_movement = gaze_vector[0] * cosine + gaze_vector[1] * sine
        y_movement = -gaze_vector[0] *  sine + gaze_vector[1] * cosine
                
        return x_movement, y_movement, gaze_vector
