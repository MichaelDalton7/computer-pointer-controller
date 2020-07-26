import cv2
import os
from arg_parser import ArgParser
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation

def main():
    arg_parser = ArgParser()
    args = arg_parser.get_args()
    print('Arguments: ', args)
    # ./face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001
    face_detection_model = FaceDetection(args.face_detection_model, args.device, args.cpu_extension)
    # ./landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009
    facial_landmarks_model = FacialLandmarksDetection(args.facial_landmark_detection_model, args.device, args.cpu_extension)
    # ./gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
    gaze_model = GazeEstimation(args.gaze_estimation_model, args.device, args.cpu_extension)
    # ./head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
    head_pose_model = HeadPoseEstimation(args.head_pose_estimation_model, args.device, args.cpu_extension)

if __name__ == '__main__':
    main()
    