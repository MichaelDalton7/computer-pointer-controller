import cv2
import os
from arg_parser import ArgParser
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from gaze_estimation import GazeEstimation
from head_pose_estimation import HeadPoseEstimation
from mouse_controller import MouseController
from input_feeder import InputFeeder

def main():
    arg_parser = ArgParser()
    args = arg_parser.get_args()

    input_file = args.input

    # If input file defined then use it else use the webcam
    if input_file:
        if not os.path.isfile(input_file):
            print("Input file cannot be found")
            exit(1)
        input_feeder = InputFeeder("video", input_file)
    else:
        input_feeder = InputFeeder("cam")

    face_detection_model = FaceDetection(args.face_detection_model, args.device, args.extensions)
    face_detection_model.load_model()

    facial_landmarks_model = FacialLandmarksDetection(args.facial_landmark_detection_model, args.device, args.extensions)
    facial_landmarks_model.load_model()

    gaze_model = GazeEstimation(args.gaze_estimation_model, args.device, args.extensions)
    gaze_model.load_model()

    head_pose_model = HeadPoseEstimation(args.head_pose_estimation_model, args.device, args.extensions)
    head_pose_model.load_model()

    mouse_controller = MouseController('medium', 'fast')

    input_feeder.load_data()

    frame_count = 0
    for ret, frame in input_feeder.next_batch():

        if not ret:
            break

        frame_count += 1

        key_pressed = cv2.waitKey(60)

        if frame_count % 5 == 0:
            cv2.imshow('Detected Face', cv2.resize(frame, (500, 500)))

        # Run inference on the face detection model
        cropped_face, face_found = face_detection_model.predict(frame.copy(), args.probability_threshold)
        
        # If no face detected get the next frame
        if not face_found:
            continue

        # Run inference on the facial landmark detection model
        left_eye, right_eye = facial_landmarks_model.predict(cropped_face.copy())

        # Run inference on the head pose estimation model
        head_pose = head_pose_model.predict(cropped_face.copy())

        # Run inference on the gaze estimation model
        # new_mouse_coordinates, gaze_vector = gaze_model.predict(left_eye, right_eye, hp_out)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cv2.destroyAllWindows()
    inputFeeder.close()


if __name__ == '__main__':
    main()
    