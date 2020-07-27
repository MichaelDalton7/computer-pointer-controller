import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IECore

'''
Class for the Head Pose Estimation Model.
'''
class HeadPoseEstimation:
    
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
        log.info("Loading Head Pose Estimation model...")
        start_time = time.time()
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        finish_time = time.time()
        log.info("Head Pose Estimation model took {} seconds to load.".format(finish_time - start_time))
        # Get Input shape
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        # Get Output name
        self.output_name = next(iter(self.model.outputs))

    def predict(self, image):
        '''
        You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image.copy())
        outputs = self.network.infer({ self.input_name : processed_image })
        return self.preprocess_output(outputs)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        return np.transpose(np.expand_dims(processed_image, axis=0), (0,3,1,2))

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        return [
            outputs['angle_y_fc'][0][0], 
            outputs['angle_p_fc'][0][0], 
            outputs['angle_r_fc'][0][0] 
        ]