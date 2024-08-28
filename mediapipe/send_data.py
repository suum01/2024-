import cv2
import mediapipe as mp
import numpy as np
from openpyxl import Workbook
# import mp_3d
import matplotlib.pyplot as plt
import math
import socket
import json

# 소켓 설정
server_ip = '192.168.1.000' # 라베파 IP 주소
server_port = 5000 # 라베파 포트 번호

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

pose_keypoints = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])

frame_counts = []
kpts_cam0=[]
kpts_camera_cam0=[]
frame_count = 0

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

save_skeleton_path = 'skeleton_cam1.mp4'

# 웹캠 열기
# cap = cv2.VideoCapture("C:/Users/User/miniconda3/envs/camera_env/0816/오른팔/이모/port_3.mp4")
cap = cv2.VideoCapture(1)

# 동영상 넓이, 높이
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# video controller
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out2 = cv2.VideoWriter(save_skeleton_path, fourcc, 20.0, (int(width), int(height)))

# 랜드마크 연결을 위한 연결 리스트 (점들을 이을 쌍)
connections = [
    (11, 13), (13, 15),  # 왼쪽 어깨-팔꿈치-손목
    (12, 14), (14, 16),  # 오른쪽 어깨-팔꿈치-손목
    (11, 12),  # 양쪽 어깨를 잇는 선
    (23, 25), (25, 27),  # 왼쪽 엉덩이-무릎-발목
    (24, 26), (26, 28),  # 오른쪽 엉덩이-무릎-발목
    (23, 24),   # 양쪽 엉덩이를 잇는 선
    (11, 23), (12, 24)  # 어깨와 엉덩이 잇는 선
]

# print('width: ',int(width),  ' height : ',int(height))

def calculateAngle(point1, point2, point3):
    
    x1, y1 = point1.x, point1.y
    x2, y2 = point2.x, point2.y
    x3, y3 = point3.x, point3.y
    
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle


def calculate3dAngle(point1, point2, point3):
    
    x1, y1, z1 = point1.x, point1.y, point1.z
    x2, y2, z2 = point2.x, point2.y, point2.z
    x3, y3, z3 = point3.x, point3.y, point3.z
    
    vector1 = np.array([x1-x2, y1-y2, z1-z2])
    vector2 = np.array([x3-x2, y3-y2, z3-z2])
    
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    angle = np.degrees(np.arccos(dot_product / (norm1 * norm2)))
    
    return angle


def classifyPose(point, output, display=False):
    labels = []
    colors = []
    
    # 점프 : 만세 (겨드랑이 각도 150도 이상)
    armpitAngle_left = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    armpitAngle_right = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    
    # 앉기 : 스쿼트 (무릎 각도 100도 이하)
    kneeAngle_left = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    kneeAngle_right = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    print('R: ', kneeAngle_right)
    print('L: ', kneeAngle_left)
    if (armpitAngle_left >= 150 and armpitAngle_right >=150):
        labels.append('jump')
    elif (kneeAngle_left <= 100 and kneeAngle_right <= 100):
        labels.append('squat')
    else:
        labels.append('none')
    
    for label in labels:
        colors.append((0, 255, 0))
        for label, color in zip(labels, colors):
            cv2.putText(output, label, (10,360), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    return output, labels
    
    
def toledmatrix(keypoints, matrix_width=64, matrix_height=64):
    scaled_keypoints = {}
    
    for name, (x,y) in keypoints.items():
        led_x = int(x*(matrix_width-1))
        led_y = int(y*(matrix_height-1))
        scaled_keypoints[name] = (led_x, led_y)
        
    return scaled_keypoints


def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()
    
landmark_indices = list(range(11, 17)) + list(range(23, 29))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    cv2.putText(image, 'FPS: {}'.format(int(frame_count)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    # 이미지 처리 및 포즈 추정
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 좌표를 내보내기 위한 코드
    frame1_keypoints = []
    frame1_kpts_camera = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        keypoints = {}
        points = {}
        for i, idx in enumerate(pose_keypoints):
            landmark = landmarks[idx]
            # sheets_landmarks[i].append([frame_count, landmark.x, landmark.y, landmark.z])
            pxl_x = landmark.x*int(width)
            pxl_y = landmark.y*int(height)
            pxl_z = landmark.z*int(width)
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            pxl_z = int(round(pxl_z))                
            points[idx] = (pxl_x, pxl_y)
            keypoints[i] = ((landmark.x, landmark.y))
            cv2.circle(image, (pxl_x, pxl_y), 8, (0, 0, 255), -1)
            
            # 점 사이 선 잇기
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx in points and end_idx in points:
                cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255),2)
    
        frame, _ = classifyPose(landmarks, image, display=False)
        
        scaled_keypoints = toledmatrix(keypoints)
        json_data = json.dumps(keypoints)
        sock.sendto(json_data.encode('utf-8'), (server_ip, server_port))
    
     #update keypoints container
    kpts_cam0.append(frame1_keypoints)
    
    cv2.imshow('Mediapipe Pose',frame)
    out2.write(image)
    
    frame_count += 1    

    # 종료 조건 설정 (ESC 키를 누르면 종료)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    # 통신
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        

# 웹캠 사용 종료
cap.release()
out2.release()
cv2.destroyAllWindows()

print('end')

# 엑셀 파일 저장
# wb.save('이모/kpts_cam1.xlsx')

write_keypoints_to_disk('kpts_cam1.dat', kpts_cam0)
