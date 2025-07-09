import cv2
import mediapipe as mp
import json
import numpy as np
from tqdm import tqdm

# --- 初始化 MediaPipe Pose ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_frame_and_extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return image, None

    landmarks_data = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        landmarks_data.append({
            'id': i,
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        })

    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

    return image, landmarks_data

def normalize_landmarks(landmarks):
    # 获取最小和最大值来进行归一化处理
    x_vals = [lm['x'] for lm in landmarks]
    y_vals = [lm['y'] for lm in landmarks]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    normalized_landmarks = []
    for lm in landmarks:
        normalized_landmarks.append({
            'id': lm['id'],
            'x': (lm['x'] - x_min) / (x_max - x_min),
            'y': (lm['y'] - y_min) / (y_max - y_min),
            'z': lm['z'],
            'visibility': lm['visibility']
        })

    return normalized_landmarks

def align_landmarks(landmarks, reference_point='hip'):
    # 获取参考点（默认是胯部，id=23）
    ref_point_idx = 23 if reference_point == 'hip' else 11  # 胯部或肩膀
    ref_point = landmarks[ref_point_idx]

    # 平移所有点，使参考点对齐到 (0.5, 0.5)
    translation_x = 0.5 - ref_point['x']
    translation_y = 0.5 - ref_point['y']

    aligned_landmarks = []
    for lm in landmarks:
        aligned_landmarks.append({
            'id': lm['id'],
            'x': lm['x'] + translation_x,
            'y': lm['y'] + translation_y,
            'z': lm['z'],
            'visibility': lm['visibility']
        })

    return aligned_landmarks

def calculate_pose_similarity(landmarks1, landmarks2):
    # 计算骨架的欧几里得距离（简单的相似性评估）
    similarity = 0
    for lm1, lm2 in zip(landmarks1, landmarks2):
        dist = np.sqrt((lm1['x'] - lm2['x'])**2 + (lm1['y'] - lm2['y'])**2)
        similarity += dist

    return similarity

def generate_video_with_pose_data(input_video_path, output_video_path, output_json_path, reference_landmarks=None):
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"错误: 无法打开视频文件 '{input_video_path}'")
    except Exception as e:
        print(e)
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_frames_data = []
    print("开始处理视频...")
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            try:
                processed_image, landmarks = process_frame_and_extract_landmarks(frame)

                if landmarks:
                    # 归一化骨架数据
                    normalized_landmarks = normalize_landmarks(landmarks)
                    # 对齐骨架数据
                    aligned_landmarks = align_landmarks(normalized_landmarks)

                    # 比较骨架与参考骨架的相似性
                    if reference_landmarks is not None:
                        similarity = calculate_pose_similarity(aligned_landmarks, reference_landmarks)
                        print(f"第 {frame_number} 帧与参考骨架的相似性: {similarity}")

                    all_frames_data.append({
                        "frame_number": frame_number,
                        "landmarks": aligned_landmarks
                    })

                out.write(processed_image)
            except Exception as e:
                print(f"处理第 {frame_number} 帧时出错: {e}")
                out.write(frame)

            pbar.update(1)
            frame_number += 1

    print("视频处理完成，正在保存数据...")

    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frames_data, f, ensure_ascii=False, indent=4)
        print(f"姿态数据成功保存到: {output_json_path}")
    except Exception as e:
        print(f"保存 JSON 文件时出错: {e}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理后的视频成功保存到: {output_video_path}")





if __name__ == "__main__":
    # 示例使用一个参考骨架数据（这里假设您已经准备了一个模板视频的骨架数据）
    REFERENCE_JSON_PATH = "pose_data.json"  # 替换为标准模板视频的 JSON 文件路径

    with open(REFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
        reference_landmarks = reference_data[0]['landmarks']  # 选择第一帧的骨架数据作为参考


#     INPUT_VIDEO_PATH = "input_video.mp4"  # 替换为您的输入视频路径
#     OUTPUT_VIDEO_PATH = "output_video_with_poses.mp4"
    INPUT_VIDEO_PATH = "camera_video.mp4"  # 替换为您的输入视频路径
    OUTPUT_VIDEO_PATH = "standard_video.mp4"
    OUTPUT_JSON_PATH = "processed_pose_data.json"

    generate_video_with_pose_data(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_JSON_PATH, reference_landmarks)

