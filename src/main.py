import cv2
import os
import datetime
import logging as log
import time
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
            log.error("Input file cannot be found")
            exit()
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
    total_face_detection_inference_time = 0
    total_facial_landmark_inference_time = 0
    total_head_pose_inference_time = 0
    total_gaze_estimation_inference_time = 0
    total_inference_time = 0
    for ret, frame in input_feeder.next_batch():

        if not ret:
            log.error("ret variable not found")
            break

        frame_count += 1

        if frame_count % args.mouse_update_interval == 0:
            cv2.imshow('Input', frame)

        key_pressed = cv2.waitKey(60)

        # Run inference on the face detection model
        start_time = time.time()
        cropped_face, face_coordinates = face_detection_model.predict(frame.copy(), args.probability_threshold)
        finish_time = time.time()
        total_face_detection_inference_time += finish_time - start_time
        total_inference_time += finish_time - start_time

        # If no face detected get the next frame
        if len(face_coordinates) == 0:
            continue

        # Run inference on the facial landmark detection model
        start_time = time.time()
        results = facial_landmarks_model.predict(cropped_face.copy())
        finish_time = time.time()
        left_eye_coordinates = results[0] 
        right_eye_coordinates = results[1] 
        left_eye_image = results[2] 
        right_eye_image = results[3] 
        left_eye_crop_coordinates = results[4] 
        right_eye_crop_coordinates = results[5] 
        total_facial_landmark_inference_time += finish_time - start_time
        total_inference_time += finish_time - start_time

        # Run inference on the head pose estimation model
        start_time = time.time()
        head_pose = head_pose_model.predict(cropped_face.copy())
        finish_time = time.time()
        total_head_pose_inference_time += finish_time - start_time
        total_inference_time += finish_time - start_time

        # Run inference on the gaze estimation model
        start_time = time.time()
        new_mouse_x_coordinate, new_mouse_y_coordinate, gaze_vector = gaze_model.predict(left_eye_image, right_eye_image, head_pose)
        finish_time = time.time()
        total_gaze_estimation_inference_time += finish_time - start_time
        total_inference_time += finish_time - start_time

        if frame_count % args.mouse_update_interval == 0:
            log.info("Mouse controller new coordinates: x = {}, y = {}".format(new_mouse_x_coordinate, new_mouse_y_coordinate))
            mouse_controller.move(new_mouse_x_coordinate, new_mouse_y_coordinate) 

            # Optional visualization configuration:
            if args.show_detected_face:
                showDetectedFace(frame, face_coordinates)

            if args.show_head_pose:
                showHeadPose(frame, head_pose)
            
            if args.show_facial_landmarks:
                showFacialLandmarks(cropped_face, left_eye_crop_coordinates, right_eye_crop_coordinates)

            if args.show_gaze_estimation:
                showGazeEstimation(frame, right_eye_coordinates, left_eye_coordinates, gaze_vector, cropped_face, face_coordinates)

        # Break if escape key pressed
        if key_pressed == 27:
            log.warning("Keyboard interrupt triggered")
            break

    # Release the capture and destroy any OpenCV windows
    cv2.destroyAllWindows()
    input_feeder.close()
    log.info("Average face detection inference time: {} seconds".format(total_face_detection_inference_time / frame_count))
    log.info("Average facial landmark detection inference time: {} seconds".format(total_facial_landmark_inference_time / frame_count))
    log.info("Average head pose estimation inference time: {} seconds".format(total_head_pose_inference_time / frame_count))
    log.info("Average gaze estimation inference time: {} seconds".format(total_gaze_estimation_inference_time / frame_count))
    log.info("Average total inference time: {} seconds".format(total_inference_time / frame_count))

def showDetectedFace(frame, face_coordinates):
    face_detected_frame = cv2.rectangle(
        frame.copy(), 
        (face_coordinates[0],face_coordinates[1]),
        (face_coordinates[2],face_coordinates[3]), 
        color=(0,0,255),
        thickness=3
    )
    cv2.imshow('Detected Face', face_detected_frame)

def showHeadPose(frame, head_pose):
    yaw_string = "Yaw: {}".format(head_pose[0])
    pitch_string = "Pitch: {}".format(head_pose[1])
    roll_string = "Roll: {}".format(head_pose[2])
    head_pose_frame = cv2.putText(
        frame.copy(), 
        yaw_string,
        org=(25,50),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.5,
        color=(0,0,255),
        thickness=3
    )
    head_pose_frame = cv2.putText(
        head_pose_frame, 
        pitch_string,
        org=(25,100),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.5,
        color=(0,0,255),
        thickness=3
    )
    head_pose_frame = cv2.putText(
        head_pose_frame, 
        roll_string,
        org=(25,150),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=1.5,
        color=(0,0,255),
        thickness=3
    )
    cv2.imshow('Head Pose', head_pose_frame)

def showFacialLandmarks(cropped_face, left_eye_crop_coordinates, right_eye_crop_coordinates):
    cropped_size = 10
    eyes_detection_frame = cv2.rectangle(
        cropped_face.copy(),
        (left_eye_crop_coordinates[0] - cropped_size, left_eye_crop_coordinates[1] - cropped_size),
        (left_eye_crop_coordinates[2] + cropped_size, left_eye_crop_coordinates[3] + cropped_size),
        color=(0,0,255), 
        thickness=3
    )
    eyes_detection_frame = cv2.rectangle(
        eyes_detection_frame,
        (right_eye_crop_coordinates[0] - cropped_size, right_eye_crop_coordinates[1] - cropped_size),
        (right_eye_crop_coordinates[2] + cropped_size, right_eye_crop_coordinates[3] + cropped_size),
        color=(0,0,255), 
        thickness=3
    )                
    cv2.imshow('Eyes Detection', cv2.resize(eyes_detection_frame, (500, 500)))

def showGazeEstimation(frame, right_eye_coordinates, left_eye_coordinates, gaze_vector, cropped_face, face_coordinates):
    right_eye_start_x = int(right_eye_coordinates[0] * cropped_face.shape[1] + face_coordinates[0])
    right_eye_start_y = int(right_eye_coordinates[1] * cropped_face.shape[0] + face_coordinates[1])
    right_eye_end_x = int(right_eye_start_x + gaze_vector[0] *  100)
    right_eye_end_y = int(right_eye_start_y - gaze_vector[1] * 100)

    left_eye_start_x = int(left_eye_coordinates[0] * cropped_face.shape[1] + face_coordinates[0])
    left_eye_start_y = int(left_eye_coordinates[1] * cropped_face.shape[0] + face_coordinates[1])
    left_eye_end_x = int(left_eye_start_x + gaze_vector[0] * 100)
    left_eye_end_y = int(left_eye_start_y - gaze_vector[1] * 100)

    gaze_estimation = cv2.arrowedLine(
        frame.copy(), 
        (right_eye_start_x, right_eye_start_y), 
        (right_eye_end_x, right_eye_end_y), 
        color=(0, 0, 255), 
        thickness=2
    )

    gaze_estimation = cv2.arrowedLine(
        gaze_estimation, 
        (left_eye_start_x, left_eye_start_y), 
        (left_eye_end_x, left_eye_end_y), 
        color=(0, 0, 255), 
        thickness=2
    )

    cv2.imshow('Gaze Estimation', gaze_estimation)


if __name__ == '__main__':
    current_time = datetime.datetime.now()
    log.basicConfig(filename='application.log',level=log.DEBUG)
    log.info("----------------Application Started {} ------------------".format(current_time.strftime("%H:%M:%S %d-%m-%Y")))
    main()
    log.info("----------------Application Finished-----------------")
    