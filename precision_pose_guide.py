import cv2
import pose_module as pm
import numpy as np
import time
import threading
import queue
import winsound

APP_WINDOW_NAME = "Precision Pose Guide"
cv2.namedWindow(APP_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(APP_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

audio_message_queue = queue.Queue()

def background_audio_worker():
    while True:
        audio_task = audio_message_queue.get()
        if audio_task is None: break
        
        if audio_task == "FLATLINE_SEQUENCE":
            winsound.Beep(1100, 5000)
            for fade_frequency in range(1100, 200, -50):
                winsound.Beep(fade_frequency, 100)
        else:
            frequency, duration = audio_task
            winsound.Beep(frequency, duration)
            
        audio_message_queue.task_done()

audio_worker_thread = threading.Thread(target=background_audio_worker, daemon=True)
audio_worker_thread.start()

video_capture = cv2.VideoCapture(0)
fitness_detector = pm.PoseDetector()

total_rep_count = 0
movement_direction_flag = 0 
active_user_feedback = "GET READY!"
selected_workout_mode = "None"
current_software_state = "MENU" 

countdown_timestamp_start = 0
rest_timer_expiry = 0
is_user_resting_status = False

def draw_centered_display_text(img, text, vertical_coordinate_y, font_scale=2, font_thickness=3, 
                               text_color=(255, 255, 255), apply_centering=False):
    text_dimensions = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)[0]
    if apply_centering:
        horizontal_coordinate_x = (img.shape[1] - text_dimensions[0]) // 2
    else:
        horizontal_coordinate_x = 50 
        
    cv2.putText(img, text, (horizontal_coordinate_x + 2, vertical_coordinate_y + 2), 
                cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), font_thickness + 2)
    cv2.putText(img, text, (horizontal_coordinate_x, vertical_coordinate_y), 
                cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, font_thickness)

while True:
    capture_success, current_frame = video_capture.read()
    if not capture_success: continue

    current_frame = cv2.resize(current_frame, (1280, 720)) 
    current_frame = fitness_detector.find_pose(current_frame, draw=False)
    pose_landmark_list = fitness_detector.find_position(current_frame, draw=False)

    if len(pose_landmark_list) != 0:
        tracking_results = fitness_detector.results
        left_arm_visibility = tracking_results.pose_landmarks.landmark[13].visibility 
        right_arm_visibility = tracking_results.pose_landmarks.landmark[14].visibility 
        target_joint_indices = (12, 14, 16) if right_arm_visibility > left_arm_visibility else (11, 13, 15)

    if current_software_state == "MENU":
        cv2.rectangle(current_frame, (0, 0), (1280, 720), (30, 30, 30), cv2.FILLED)
        draw_centered_display_text(current_frame, "PRECISION POSE GUIDE", 100, font_scale=3, 
                                   font_thickness=4, apply_centering=True)
        draw_centered_display_text(current_frame, "[1] BICEP CURL  [2] ROW  [3] PRESS", 220, 
                                   font_scale=1.5, text_color=(0, 255, 0), apply_centering=True)
                                   
        cv2.rectangle(current_frame, (180, 320), (1100, 650), (50, 50, 50), cv2.FILLED)
        cv2.rectangle(current_frame, (180, 320), (1100, 650), (200, 200, 200), 4)
        
        draw_centered_display_text(current_frame, "--- TEACHER / EXAMINER GUIDE ---", 380, font_scale=1.2, text_color=(0, 255, 255), apply_centering=True)
        draw_centered_display_text(current_frame, "* PRESS [1], [2], OR [3] TO CHOOSE A WORKOUT", 450, font_scale=1, text_color=(255, 255, 255), apply_centering=True)
        draw_centered_display_text(current_frame, "* PRESS [ENTER] ON THE GUIDE SCREEN TO START", 510, font_scale=1, text_color=(255, 255, 255), apply_centering=True)
        draw_centered_display_text(current_frame, "* PRESS [R] DURING WORKOUT TO RESET REPS", 570, font_scale=1, text_color=(255, 255, 255), apply_centering=True)
        draw_centered_display_text(current_frame, "* PRESS [ESC] ANYTIME TO CANCEL OR SKIP REST", 630, font_scale=1, text_color=(255, 255, 255), apply_centering=True)
        
        draw_centered_display_text(current_frame, "[Q] QUIT APPLICATION", 690, font_scale=1.2, text_color=(0, 0, 255), apply_centering=True)

    elif current_software_state == "GUIDE":
        cv2.rectangle(current_frame, (0, 0), (1280, 720), (50, 50, 50), cv2.FILLED)
        
        if selected_workout_mode == "Curl": workout_instructions = "CURL: Keep elbow pinned. Squeeze up!"
        elif selected_workout_mode == "Row": workout_instructions = "ROW: Back flat. Pull dumbbell to hip."
        elif selected_workout_mode == "Press": workout_instructions = "PRESS: Push weight straight up to lock out!"
        
        draw_centered_display_text(current_frame, f"GUIDE: {selected_workout_mode.upper()}", 200, font_scale=3, apply_centering=True)
        draw_centered_display_text(current_frame, workout_instructions, 350, font_scale=1.3, 
                                   text_color=(200, 255, 200), apply_centering=True)
        draw_centered_display_text(current_frame, "PRESS [ENTER] TO PROCEED", 550, font_scale=2, 
                                   text_color=(0, 255, 255), apply_centering=True)

    elif current_software_state == "COUNTDOWN":
        remaining_seconds = int(10 - (time.time() - countdown_timestamp_start))
        cv2.rectangle(current_frame, (0, 0), (1280, 720), (0, 30, 0), cv2.FILLED)
        draw_centered_display_text(current_frame, "GET IN POSITION!", 250, font_scale=3, apply_centering=True)
        draw_centered_display_text(current_frame, str(remaining_seconds), 450, font_scale=6, 
                                   text_color=(0, 255, 255), apply_centering=True)
        if remaining_seconds <= 0: current_software_state = "WORKOUT"

    elif current_software_state == "WORKOUT":
        if is_user_resting_status:
            rest_timer_value = int(rest_timer_expiry - time.time())
            cv2.rectangle(current_frame, (0, 0), (1280, 720), (50, 20, 0), cv2.FILLED)
            draw_centered_display_text(current_frame, f"REST: {rest_timer_value//60}:{rest_timer_value%60:02d}", 
                                       400, font_scale=4, text_color=(0, 255, 255), apply_centering=True)
            draw_centered_display_text(current_frame, "PRESS [ESC] TO SKIP REST", 600, font_scale=1.5, 
                                       text_color=(200, 200, 200), apply_centering=True)
            if rest_timer_value <= 0: is_user_resting_status = False

        elif len(pose_landmark_list) != 0:
            current_joint_angle = fitness_detector.calculate_joint_angle(current_frame, target_joint_indices[0], 
                                                                        target_joint_indices[1], target_joint_indices[2], 
                                                                        pose_landmark_list)
            if selected_workout_mode == "Curl":
                rep_completion_percentage = np.interp(current_joint_angle, (65, 165), (100, 0))
                visual_progress_bar_y = np.interp(current_joint_angle, (65, 165), (100, 650))
            elif selected_workout_mode == "Row":
                rep_completion_percentage = np.interp(current_joint_angle, (85, 155), (100, 0))
                visual_progress_bar_y = np.interp(current_joint_angle, (85, 155), (100, 650))
            elif selected_workout_mode == "Press":
                rep_completion_percentage = np.interp(current_joint_angle, (200, 290), (100, 0))
                visual_progress_bar_y = np.interp(current_joint_angle, (200, 290), (100, 650))

            ui_display_color = (0, 0, 255)
            
            if total_rep_count < 12:
                if rep_completion_percentage == 100:
                    ui_display_color = (0, 255, 0)
                    active_user_feedback = "EXCELLENT!"
                    if movement_direction_flag == 0:
                        total_rep_count += 0.5
                        movement_direction_flag = 1
                        if total_rep_count < 12: audio_message_queue.put((1500, 200)) 

                if rep_completion_percentage == 0:
                    ui_display_color = (0, 255, 0)
                    active_user_feedback = "GO!"
                    if movement_direction_flag == 1:
                        total_rep_count += 0.5
                        movement_direction_flag = 0

            if total_rep_count >= 12:
                cv2.rectangle(current_frame, (1100, 100), (1175, 650), (0, 255, 0), cv2.FILLED)
                draw_centered_display_text(current_frame, "REPS: 12 / 12", 100, font_scale=4)
                draw_centered_display_text(current_frame, "SET COMPLETE!", 180, font_scale=1.5, text_color=(0, 255, 0))
                cv2.imshow(APP_WINDOW_NAME, current_frame)
                cv2.waitKey(1)
                
                time.sleep(0.8)
                audio_message_queue.put("FLATLINE_SEQUENCE")
                is_user_resting_status = True
                rest_timer_expiry = time.time() + (3 * 60)
                total_rep_count = 0 

            cv2.rectangle(current_frame, (1100, 100), (1175, 650), ui_display_color, 4)
            cv2.rectangle(current_frame, (1100, int(visual_progress_bar_y)), (1175, 650), ui_display_color, cv2.FILLED)
            draw_centered_display_text(current_frame, f"REPS: {int(total_rep_count)} / 12", 100, font_scale=4)
            draw_centered_display_text(current_frame, active_user_feedback, 180, font_scale=1.5, text_color=ui_display_color)
            draw_centered_display_text(current_frame, "[R] RESET | [ESC] MENU", 690, font_scale=0.8)

    cv2.imshow(APP_WINDOW_NAME, current_frame)
    keyboard_input_key = cv2.waitKey(1) & 0xFF
    
    if keyboard_input_key == ord('q'): break

    if current_software_state == "MENU":
        if keyboard_input_key == ord('1'): selected_workout_mode = "Curl"; current_software_state = "GUIDE"
        elif keyboard_input_key == ord('2'): selected_workout_mode = "Row"; current_software_state = "GUIDE"
        elif keyboard_input_key == ord('3'): selected_workout_mode = "Press"; current_software_state = "GUIDE"

    elif current_software_state == "GUIDE" and keyboard_input_key == 13: 
        current_software_state = "COUNTDOWN"; countdown_timestamp_start = time.time()

    elif keyboard_input_key == 27: 
        current_software_state = "MENU"; is_user_resting_status = False; total_rep_count = 0

    elif current_software_state == "WORKOUT":
        if keyboard_input_key == ord('r'): total_rep_count = 0 

video_capture.release()
cv2.destroyAllWindows()