import cv2
import logging as log
import time
import numpy as np
from openvino.inference_engine import IECore

'''
Class for the Face Landmarks Detection Model.
'''
class FacialLandmarksDetection:

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
        log.info("Loading Facial Landmarks Detection model...")
        start_time = time.time()
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        finish_time = time.time()
        log.info("Facial Landmarks Detection model took {} seconds to load.".format(finish_time - start_time))
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
        outputs = self.network.infer({self.input_name : processed_image})
        return self.preprocess_output(outputs, image.copy())

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_image = cv2.resize(processed_image, (self.input_shape[3], self.input_shape[2]))
        return np.transpose(np.expand_dims(processed_image, axis=0), (0, 3, 1, 2))

    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        image_height = image.shape[0]
        image_width = image.shape[1]

        detections = outputs[self.output_name][0]
                
        left_eye_x = detections[0][0][0]
        left_eye_y = detections[1][0][0]
        right_eye_x = detections[2][0][0]
        right_eye_y = detections[3][0][0]

        cropped_size = 10
        left_eye_x_min = int(left_eye_x * image_width) - cropped_size
        left_eye_y_min = int(left_eye_y * image_height) - cropped_size
        left_eye_x_max = int(left_eye_x * image_width) + cropped_size
        left_eye_y_max = int(left_eye_y * image_height) + cropped_size

        right_eye_x_min = int(right_eye_x * image_width) - cropped_size
        right_eye_y_min = int(right_eye_y * image_height) - cropped_size
        right_eye_x_max = int(right_eye_x * image_width) + cropped_size
        right_eye_y_max = int(right_eye_y * image_height) + cropped_size
        
        left_eye_crop_coordinates = (left_eye_x_min, left_eye_y_min, left_eye_x_max, left_eye_y_max)
        right_eye_crop_coordinates = (right_eye_x_min, right_eye_y_min, right_eye_x_max, right_eye_y_max)

        # Return left and right eye images
        return (
            (left_eye_x, left_eye_y),
            (right_eye_x, right_eye_y),
            image[left_eye_y_min:left_eye_y_max, left_eye_x_min:left_eye_x_max], 
            image[right_eye_y_min:right_eye_y_max, right_eye_x_min:right_eye_x_max],
            left_eye_crop_coordinates,
            right_eye_crop_coordinates,
        )
