from argparse import ArgumentParser

i_desc = '''The path to the input video file. If no input file path is specified then 
            the application will try to use the video camera.'''
d_desc = '''The device type that inference should be run on. 
            If not value is specifed the CPU will be used.'''
fm_desc = '''The path to the face detection model and it's weights file (.xml and .bin files). 
             This path should include the model name but not the file extension'''
flm_desc = '''The path to the facial landmark detection model and it's weights file (.xml and .bin files). 
              This path should include the model name but not the file extension'''
gm_desc = '''The path to the gaze estimation model and it's weights file (.xml and .bin files). 
              This path should include the model name but not the file extension'''
hpm_desc = '''The path to the head pose estimation model and it's weights file (.xml and .bin files). 
              This path should include the model name but not the file extension'''
prob_desc = '''The probability threshold to be used to separate inference results that are correct from the 
               ones that are incorrect. This should be a floating point value between 0 and 1. 
               The default value is 0.6'''
ext_desc = '''A path to an extension to be used by OpenVino'''
mui_desc = '''The number of frames between each mouse update. The default value is 5.'''
show_detected_face_desc = ''' '''
show_facial_landmarks_desc = ''' '''
show_head_pose_desc = ''' '''
show_gaze_estimation_desc = ''' '''

class ArgParser:
    def __init__(self):
        '''
        Constructor for the ArgParser class
        '''
        self.parser = ArgumentParser("Run inference on an input or video stream to control the devices mouse with eye gaze")

        # Add parser groups
        self.parser._action_groups.pop()
        self.required = self.parser.add_argument_group('Required arguments')
        self.optional = self.parser.add_argument_group('Optional arguments')

        # Required Arguments
        self.required.add_argument("-fm", "--face-detection-model", required=True, type=str, help=fm_desc)
        self.required.add_argument("-flm", "--facial-landmark-detection-model", required=True, type=str, help=flm_desc)
        self.required.add_argument("-gm", "--gaze-estimation-model", required=True, type=str, help=gm_desc)
        self.required.add_argument("-hpm", "--head-pose-estimation-model", required=True, type=str, help=hpm_desc)

        # Optional Arguments
        self.optional.add_argument("-i", "--input", required=False, type=str, help=i_desc)
        self.optional.add_argument("-d", "--device", required=False, type=str, default="CPU", help=d_desc)
        self.optional.add_argument("-pr", "--probability-threshold", required=False, type=float, default=0.6, help=prob_desc)
        self.optional.add_argument("-ext", "--extensions", required=False, type=str, default=None, help=ext_desc)
        self.optional.add_argument("-mui", "--mouse-update-interval", required=False, type=int, default=5, help=mui_desc)
        self.optional.add_argument("--show-detected-face", required=False, type=bool, default=False, help=show_detected_face_desc)
        self.optional.add_argument("--show-facial-landmarks", required=False, type=bool, default=False, help=show_facial_landmarks_desc)
        self.optional.add_argument("--show-head-pose", required=False, type=bool, default=False, help=show_head_pose_desc)
        self.optional.add_argument("--show-gaze-estimation", required=False, type=bool, default=False, help=show_gaze_estimation_desc)
    
    def get_args(self):
        '''
        Gets the arguments from the command line.
        '''
        return self.parser.parse_args()


    def get_parser(self):
        '''
        Gets the parser which can be used to parse command line arguments
        '''
        return self.parser

