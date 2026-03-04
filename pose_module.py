import cv2
import mediapipe as mp
import math

class PoseDetector():

    def __init__(self, mode=False, model_complexity=1, smooth=True, 
                 enable_segmentation=False, smooth_segmentation=True, 
                 detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.model_complexity, self.smooth, 
                                      self.enable_segmentation, self.smooth_segmentation, 
                                      self.detection_con, self.track_con)

    def find_pose(self, frame_img, draw=True):
        img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(frame_img, self.results.pose_landmarks, 
                                            self.mp_pose.POSE_CONNECTIONS)
        return frame_img

    def find_position(self, frame_img, draw=True):
        landmark_data_list = []
        if self.results.pose_landmarks:
            for landmark_id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, channels = frame_img.shape
                center_coordinate_x, center_coordinate_y = int(landmark.x * width), int(landmark.y * height)
                landmark_data_list.append([landmark_id, center_coordinate_x, center_coordinate_y])
                if draw:
                    cv2.circle(frame_img, (center_coordinate_x, center_coordinate_y), 5, (255, 0, 0), cv2.FILLED)
        return landmark_data_list

    def calculate_joint_angle(self, frame_img, shoulder_idx, elbow_idx, wrist_idx, landmark_data_list, draw=True):
        x_shoulder, y_shoulder = landmark_data_list[shoulder_idx][1], landmark_data_list[shoulder_idx][2]
        x_elbow, y_elbow = landmark_data_list[elbow_idx][1], landmark_data_list[elbow_idx][2]
        x_wrist, y_wrist = landmark_data_list[wrist_idx][1], landmark_data_list[wrist_idx][2]
        angle_result = math.degrees(math.atan2(y_wrist - y_elbow, x_wrist - x_elbow) - 
                                     math.atan2(y_shoulder - y_elbow, x_shoulder - x_elbow))
        if angle_result < 0:
            angle_result += 360
        if draw:
            cv2.line(frame_img, (x_shoulder, y_shoulder), (x_elbow, y_elbow), (255, 255, 255), 3)
            cv2.line(frame_img, (x_wrist, y_wrist), (x_elbow, y_elbow), (255, 255, 255), 3)
            cv2.circle(frame_img, (x_shoulder, y_shoulder), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame_img, (x_elbow, y_elbow), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame_img, (x_wrist, y_wrist), 10, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame_img, str(int(angle_result)), (x_elbow - 50, y_elbow + 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return angle_result