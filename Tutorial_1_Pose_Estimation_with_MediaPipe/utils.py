import csv
import os
from os.path import isfile, join
import cv2
import mediapipe as mp
import numpy as np


def define_landmark_columns():
    """
    # This function creates a list of column names based on the specified markers and their corresponding positions (X, Y, Z, visibility). It also allows for the creation of world coordinate columns for body landmarks.
    """
    markers_body = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW',
        'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX',
        'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]

# Define landmark names for hands (42 landmarks - 21 per hand)
    markers_hands = [
        'LEFT_WRIST_HAND', 'LEFT_THUMB_CMC', 'LEFT_THUMB_MCP', 'LEFT_THUMB_IP', 'LEFT_THUMB_TIP', 'LEFT_INDEX_FINGER_MCP',
        'LEFT_INDEX_FINGER_PIP', 'LEFT_INDEX_FINGER_DIP', 'LEFT_INDEX_FINGER_TIP', 'LEFT_MIDDLE_FINGER_MCP',
        'LEFT_MIDDLE_FINGER_PIP', 'LEFT_MIDDLE_FINGER_DIP', 'LEFT_MIDDLE_FINGER_TIP', 'LEFT_RING_FINGER_MCP',
        'LEFT_RING_FINGER_PIP', 'LEFT_RING_FINGER_DIP', 'LEFT_RING_FINGER_TIP', 'LEFT_PINKY_MCP',
        'LEFT_PINKY_PIP', 'LEFT_PINKY_DIP', 'LEFT_PINKY_TIP',
        'RIGHT_WRIST_HAND', 'RIGHT_THUMB_CMC', 'RIGHT_THUMB_MCP', 'RIGHT_THUMB_IP', 'RIGHT_THUMB_TIP', 'RIGHT_INDEX_FINGER_MCP',
        'RIGHT_INDEX_FINGER_PIP', 'RIGHT_INDEX_FINGER_DIP', 'RIGHT_INDEX_FINGER_TIP', 'RIGHT_MIDDLE_FINGER_MCP',
        'RIGHT_MIDDLE_FINGER_PIP', 'RIGHT_MIDDLE_FINGER_DIP', 'RIGHT_MIDDLE_FINGER_TIP', 'RIGHT_RING_FINGER_MCP',
        'RIGHT_RING_FINGER_PIP', 'RIGHT_RING_FINGER_DIP', 'RIGHT_RING_FINGER_TIP', 'RIGHT_PINKY_MCP',
        'RIGHT_PINKY_PIP', 'RIGHT_PINKY_DIP', 'RIGHT_PINKY_TIP'
    ]

    markers_face = [str(index) for index in range(478)]

    def create_columns(markers, positions, suffix=''):
        columns = ['time']
        for marker in markers:
            for position in positions:
                columns.append(f"{position}_{marker}{suffix}")
        return columns

    columns_body = create_columns(
        markers_body,
        ['X', 'Y', 'Z', 'visibility']
    )

    columns_body_world = create_columns(
        markers_body,
        ['X', 'Y', 'Z', 'visibility'],
        '_WORLD'
    )

    columns_hands = create_columns(
        markers_hands,
        ['X', 'Y', 'Z']
    )

    columns_face = create_columns(
        markers_face,
        ['X', 'Y', 'Z']
    )

    # print(f"Body landmarks: {len(markers_body)}")
    # print(f"Hand landmarks: {len(markers_hands)}")
    # print(f"Face landmarks: {len(markers_face)}")
    # print(f"\nTotal CSV columns:")
    # print(f"  Body CSV: {len(columns_body)} columns")
    # print(f"  Hands CSV: {len(columns_hands)} columns")
    # print(f"  Face CSV: {len(columns_face)} columns")

    return (columns_body,
            columns_body_world,
            columns_hands,
            columns_face,
            )


# Function for extracting position data
def extract_positions(landmarks, include_visibility=True):
    """Extract numerical values from MediaPipe landmarks"""

    positions = []

    for lm in landmarks.landmark:
        positions.extend([lm.x, lm.y, lm.z])

        if include_visibility:
            positions.append(lm.visibility)

    return positions


def run_mediapipe(input_folder, output_folder, mp_options, mp_model_options):
    """
    Extract MediaPipe landmarks and save one CSV per input video.
    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to the folder where output CSV files will be saved.
        mp_options (dict): Dictionary containing options for MediaPipe processing.
        mp_model_options (dict): Dictionary containing options for the MediaPipe model.
    """

    ######### Configure folders and MediaPipe options #########
    # Select webcam input or list all video files in the input folder
    webcam_mode = mp_options.get("webcam_mode", False)
    if webcam_mode:
        video_files = ['webcam_capture']
    else:
        video_files = [f for f in os.listdir(input_folder) if isfile(
            join(input_folder, f)) and not f.startswith('.')]

    # Configure the output folder for motion-tracking CSV files
    csv_folder = os.path.join(output_folder, "output_timeseries")
    os.makedirs(csv_folder, exist_ok=True)

    # Configure the output folder for videos with skeleton overlay
    output_folder_videos = os.path.join(output_folder, "output_videos")
    os.makedirs(output_folder_videos, exist_ok=True)

    # Define landmark columns for body, hands, and face
    columns_body, columns_body_world, columns_hands, columns_face = define_landmark_columns()

    # Initialize MediaPipe Holistic model
    mp_holistic = mp.solutions.holistic

    # Configure MediaPipe drawing utilities for overlay videos
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    ################ Run the main processing loop ################
    # ----- Process each video or the webcam stream -----
    for video_idx, video_file in enumerate(video_files, 1):
        base_filename = os.path.splitext(video_file)[0]
        output_path = os.path.join(csv_folder, f'{base_filename}.csv')

        # Skip if the output CSV already exists for this video
        if webcam_mode == False and os.path.exists(output_path):
            print(f"\n{'=' * 60}")
            print(
                f"Skipping video {video_idx}/{len(video_files)}: {video_file} (CSV already exists)")
            print(f"{'=' * 60}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing video {video_idx}/{len(video_files)}: {video_file}")
        print(f"{'=' * 60}")

        # Open video or webcam
        video_path = None if webcam_mode else os.path.join(
            input_folder, video_file)
        capture = cv2.VideoCapture(0 if webcam_mode else video_path)

        # Get video properties
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(
            cv2.CAP_PROP_FRAME_COUNT)) if not webcam_mode else 0

        if not capture.isOpened():
            raise RuntimeError(
                "Could not open webcam." if webcam_mode
                else f"Could not open video: {video_path}"
            )

        # Webcam drivers may not report a usable FPS; use a common fallback.
        if fps <= 0:
            fps = 30.0

        print("Video properties:")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(
            f"  Total frames: {total_frames if not webcam_mode else 'live webcam'}")

        # Construct the unified CSV header based on the selected options
        full_header = ['Time']
        if mp_options['body_csv']:
            full_header.extend(columns_body[1:])
        if mp_options['body_world_csv']:
            full_header.extend(columns_body_world[1:])
        if mp_options['hands_csv']:
            full_header.extend(columns_hands[1:])
        if mp_options['face_csv']:
            full_header.extend(columns_face[1:])

        # Initialize storage with the header row
        rows = [full_header]
        timestamp = 0

        # Configure optional overlay-video output
        overlay_writer = None
        if mp_options.get('save_overlay_video', False):
            overlay_filename = (
                'webcam_capture_overlay.mp4' if webcam_mode
                else f'{os.path.splitext(video_file)[0]}_overlay.mp4'
            )
            overlay_path = os.path.join(output_folder_videos, overlay_filename)
            overlay_writer = cv2.VideoWriter(
                overlay_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

        # --- Process video with MediaPipe Holistic ---
        with mp_holistic.Holistic(**mp_model_options) as holistic:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                # Convert BGR to RGB before processing the frame
                results = holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Start the current row with the timestamp
                row = [timestamp]

                # -- Extract and append landmark data based on selected options --
                # Body data
                if mp_options['body_csv']:
                    body_data = (
                        extract_positions(results.pose_landmarks)
                        if results.pose_landmarks
                        else [np.nan] * (len(columns_body) - 1)
                    )
                    row.extend(body_data)

                # Body world-coordinate data
                if mp_options['body_world_csv']:
                    world_data = (
                        extract_positions(results.pose_world_landmarks)
                        if results.pose_world_landmarks
                        else [np.nan] * (len(columns_body_world) - 1)
                    )
                    row.extend(world_data)

                # Hand data
                if mp_options['hands_csv']:
                    left_hand = (
                        extract_positions(results.left_hand_landmarks, False)
                        if results.left_hand_landmarks else [np.nan] * 63
                    )
                    right_hand = (
                        extract_positions(results.right_hand_landmarks, False)
                        if results.right_hand_landmarks else [np.nan] * 63
                    )
                    row.extend(left_hand + right_hand)

                # Face data
                if mp_options['face_csv']:
                    face_data = (
                        extract_positions(results.face_landmarks, False)
                        if results.face_landmarks
                        else [np.nan] * (len(columns_face) - 1)
                    )
                    row.extend(face_data)

                # Append the completed row and increment the timestamp
                rows.append(row)
                timestamp += 1000 / fps

                # -- Draw selected landmarks on the output frame --
                if mp_options['show_overlay']:
                    if mp_options['body_csv'] and results.pose_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
                        )
                    if mp_options['hands_csv']:
                        if results.left_hand_landmarks:
                            drawing_utils.draw_landmarks(
                                frame,
                                results.left_hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                drawing_styles.get_default_hand_landmarks_style(),
                                drawing_styles.get_default_hand_connections_style()
                            )
                        if results.right_hand_landmarks:
                            drawing_utils.draw_landmarks(
                                frame,
                                results.right_hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                drawing_styles.get_default_hand_landmarks_style(),
                                drawing_styles.get_default_hand_connections_style()
                            )
                    if mp_options['face_csv'] and results.face_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            results.face_landmarks,
                            mp_holistic.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    overlay_writer.write(frame)

                if mp_options['show_overlay']:
                    cv2.imshow('MediaPipe Holistic', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                        break

        # --- Release video capture ---
        capture.release()
        if overlay_writer is not None:
            overlay_writer.release()
            print(f"Saved overlay video: {overlay_path}")
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # this is needed to close the window for mac users

        # --- Save one CSV file for the current video ---
        with open(output_path, 'w', newline='') as file:
            csv.writer(file).writerows(rows)

        print(f"Saved: {output_path} ({total_frames} frames)")

    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"All files saved to: {os.path.abspath(csv_folder)}")
    print(f"{'='*60}")
