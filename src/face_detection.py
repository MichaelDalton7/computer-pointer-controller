import cv2
import logging as log
import time
import numpy as np
from openvino.inference_engine import IECore

'''
Class for the Face Detection Model.
'''
class FaceDetection:
   
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
        self.model = None
        self.network = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None

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
        log.info("Loading Face Detection model...")
        start_time = time.time()
        self.network = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        finish_time = time.time()
        log.info("Face Detection model took {} seconds to load.".format(finish_time - start_time))
        # Get Input shape
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        # Get Output name
        self.output_name = next(iter(self.model.outputs))

    def predict(self, image, probability_threshold):
        '''
        You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image.copy())
        outputs = self.network.infer({self.input_name : processed_image})
        return self.preprocess_output(outputs, image.copy(), probability_threshold)

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        return np.transpose(np.expand_dims(processed_image, axis=0), (0, 3, 1, 2))

    def preprocess_output(self, outputs, image, probability_threshold):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        face_coordinates = []
        detected_faces = outputs[self.output_name][0][0]
        for detection in detected_faces:
            confidence = detection[2]
            
            if confidence >= probability_threshold:
                xmin = detection[3]
                ymin = detection[4]
                xmax = detection[5]
                ymax = detection[6]
            
                face_coordinates = [xmin, ymin, xmax, ymax]
                break

        # If no face detected return False values
        if (len(face_coordinates) == 0):
            return [], []

        image_height = image.shape[0]
        image_width = image.shape[1]
        face_coordinates = face_coordinates* np.array([image_width, image_height, image_width, image_height])
        face_coordinates = face_coordinates.astype(np.int32)
        
        return image[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]], face_coordinates
