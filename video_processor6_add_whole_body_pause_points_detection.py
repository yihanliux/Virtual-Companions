import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from video_processor2 import process_frame_and_extract_landmarks, normalize_landmarks, align_landmarks, \
    calculate_pose_similarity


def calculate_foot_tip_height(landmarks, foot='right'):
    """
    计算右脚或左脚脚尖的y坐标高度
    :param landmarks: 当前帧的骨架数据
    :param foot: 'right' 或 'left'
    :return: 脚尖的y坐标
    """
    foot_tip_idx = 32 if foot == 'right' else 28  # 右脚或左脚脚尖的索引
    foot_tip = landmarks[foot_tip_idx]
    return foot_tip['y']  # 返回脚尖的y坐标（垂直方向）


def detect_pause_points(landmarks_data, foot='right', threshold=0.01, min_pause_duration=5):
    """
    根据脚尖的高度变化检测停顿点，即前后动作之间的停顿
    :param landmarks_data: 所有帧的骨架数据
    :param foot: 'right' 或 'left'
    :param threshold: 脚尖高度变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_foot_height = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        current_foot_height = calculate_foot_tip_height(frame_data['landmarks'], foot)

        # 如果是第一次计算脚尖高度，初始化
        if previous_foot_height is None:
            previous_foot_height = current_foot_height
            continue

        # 判断当前脚尖高度与前一帧的变化是否小于阈值（即进入停顿状态）
        if abs(current_foot_height - previous_foot_height) < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_foot_height = current_foot_height

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def calculate_hip_coordinates(landmarks, side='left'):
    """
    计算左胯部或右胯部的x, y, z坐标
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 胯部的x, y, z坐标（返回为一个tuple）
    """
    # 左胯部的索引是23，右胯部是24
    hip_idx = 23 if side == 'left' else 24
    hip = landmarks[hip_idx]
    return (hip['x'], hip['y'], hip['z'])  # 返回胯部的x, y, z坐标


def detect_hip_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据胯部的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 胯部坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_hip_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的胯部坐标（x, y, z）
        current_hip_coordinates = calculate_hip_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算胯部坐标，初始化
        if previous_hip_coordinates is None:
            previous_hip_coordinates = current_hip_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_hip_coordinates[0] - previous_hip_coordinates[0])
        delta_y = abs(current_hip_coordinates[1] - previous_hip_coordinates[1])
        delta_z = abs(current_hip_coordinates[2] - previous_hip_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_hip_coordinates = current_hip_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def calculate_knee_coordinates(landmarks, side='left'):
    """
    计算左膝盖或右膝盖的x, y, z坐标
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 膝盖的x, y, z坐标（返回为一个tuple）
    """
    # 左膝盖的索引是27，右膝盖是28
    knee_idx = 27 if side == 'left' else 28
    knee = landmarks[knee_idx]
    return (knee['x'], knee['y'], knee['z'])  # 返回膝盖的x, y, z坐标


def detect_knee_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据膝盖的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 膝盖坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_knee_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的膝盖坐标（x, y, z）
        current_knee_coordinates = calculate_knee_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算膝盖坐标，初始化
        if previous_knee_coordinates is None:
            previous_knee_coordinates = current_knee_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_knee_coordinates[0] - previous_knee_coordinates[0])
        delta_y = abs(current_knee_coordinates[1] - previous_knee_coordinates[1])
        delta_z = abs(current_knee_coordinates[2] - previous_knee_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_knee_coordinates = current_knee_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def calculate_shoulder_coordinates(landmarks, side='left'):
    """
    计算左肩膀或右肩膀的x, y, z坐标
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 肩膀的x, y, z坐标（返回为一个tuple）
    """
    # 左肩膀的索引是11，右肩膀是12
    shoulder_idx = 11 if side == 'left' else 12
    shoulder = landmarks[shoulder_idx]
    return (shoulder['x'], shoulder['y'], shoulder['z'])  # 返回肩膀的x, y, z坐标


def detect_shoulder_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据肩膀的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 肩膀坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_shoulder_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的肩膀坐标（x, y, z）
        current_shoulder_coordinates = calculate_shoulder_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算肩膀坐标，初始化
        if previous_shoulder_coordinates is None:
            previous_shoulder_coordinates = current_shoulder_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_shoulder_coordinates[0] - previous_shoulder_coordinates[0])
        delta_y = abs(current_shoulder_coordinates[1] - previous_shoulder_coordinates[1])
        delta_z = abs(current_shoulder_coordinates[2] - previous_shoulder_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_shoulder_coordinates = current_shoulder_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def calculate_arm_coordinates(landmarks, side='left'):
    """
    计算左臂或右臂的x, y, z坐标（肩膀到肘部）
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 左臂或右臂的x, y, z坐标
    """
    # 左臂的关键点是13（左肩膀）到15（左肘部），右臂是14（右肩膀）到16（右肘部）
    arm_idx = 13 if side == 'left' else 14  # 左臂13，右臂14（肩膀位置）
    arm = landmarks[arm_idx]
    return (arm['x'], arm['y'], arm['z'])  # 返回臂部的x, y, z坐标


def calculate_elbow_coordinates(landmarks, side='left'):
    """
    计算左肘部或右肘部的x, y, z坐标
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 肘部的x, y, z坐标
    """
    # 左肘部的关键点是15，右肘部是16
    elbow_idx = 15 if side == 'left' else 16
    elbow = landmarks[elbow_idx]
    return (elbow['x'], elbow['y'], elbow['z'])  # 返回肘部的x, y, z坐标


def calculate_hand_coordinates(landmarks, side='left'):
    """
    计算左手掌或右手掌的x, y, z坐标
    :param landmarks: 当前帧的骨架数据
    :param side: 'left' 或 'right'
    :return: 手掌的x, y, z坐标
    """
    # 左手掌的关键点是17，右手掌是18
    hand_idx = 17 if side == 'left' else 18
    hand = landmarks[hand_idx]
    return (hand['x'], hand['y'], hand['z'])  # 返回手掌的x, y, z坐标


def detect_arm_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据左臂或右臂的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 臂部坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_arm_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的臂部坐标（x, y, z）
        current_arm_coordinates = calculate_arm_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算臂部坐标，初始化
        if previous_arm_coordinates is None:
            previous_arm_coordinates = current_arm_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_arm_coordinates[0] - previous_arm_coordinates[0])
        delta_y = abs(current_arm_coordinates[1] - previous_arm_coordinates[1])
        delta_z = abs(current_arm_coordinates[2] - previous_arm_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_arm_coordinates = current_arm_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def detect_elbow_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据左肘部或右肘部的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 肘部坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_elbow_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的肘部坐标（x, y, z）
        current_elbow_coordinates = calculate_elbow_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算肘部坐标，初始化
        if previous_elbow_coordinates is None:
            previous_elbow_coordinates = current_elbow_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_elbow_coordinates[0] - previous_elbow_coordinates[0])
        delta_y = abs(current_elbow_coordinates[1] - previous_elbow_coordinates[1])
        delta_z = abs(current_elbow_coordinates[2] - previous_elbow_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_elbow_coordinates = current_elbow_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def detect_hand_pause_points(landmarks_data, side='left', threshold=0.01, min_pause_duration=5):
    """
    根据左手掌或右手掌的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param side: 'left' 或 'right'
    :param threshold: 手掌坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_hand_coordinates = None
    is_pausing = False

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的手掌坐标（x, y, z）
        current_hand_coordinates = calculate_hand_coordinates(frame_data['landmarks'], side)

        # 如果是第一次计算手掌坐标，初始化
        if previous_hand_coordinates is None:
            previous_hand_coordinates = current_hand_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_hand_coordinates[0] - previous_hand_coordinates[0])
        delta_y = abs(current_hand_coordinates[1] - previous_hand_coordinates[1])
        delta_z = abs(current_hand_coordinates[2] - previous_hand_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_hand_coordinates = current_hand_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points


def calculate_face_coordinates(landmarks, point_indices):
    """
    计算面部某些关键点的x, y, z坐标的平均值
    :param landmarks: 当前帧的骨架数据
    :param point_indices: 面部关键点的索引列表
    :return: 面部关键点的x, y, z坐标（返回为一个tuple）
    """
    # Ensure landmarks are available for all specified indices
    try:
        x_coords = [landmarks[idx].x for idx in point_indices if idx < len(landmarks)]
        y_coords = [landmarks[idx].y for idx in point_indices if idx < len(landmarks)]
        z_coords = [landmarks[idx].z for idx in point_indices if idx < len(landmarks)]

        if not x_coords or not y_coords or not z_coords:
            raise ValueError("Missing landmarks for one or more points.")

        return (np.mean(x_coords), np.mean(y_coords), np.mean(z_coords))  # 返回面部关键点的x, y, z坐标的平均值
    except Exception as e:
        print(f"Error calculating face coordinates: {e}")
        return (0.0, 0.0, 0.0)  # Return a default value if any issue occurs


def detect_face_pause_points(landmarks_data, threshold=0.01, min_pause_duration=5):
    """
    根据面部关键点的x, y, z坐标变化检测停顿点
    :param landmarks_data: 所有帧的骨架数据
    :param threshold: 面部关键点坐标变化的阈值，用于判断是否进入停顿状态
    :param min_pause_duration: 最小停顿持续帧数，表示必须持续几帧才能判定为停顿
    :return: 停顿点的索引列表
    """
    pause_points = []
    pause_duration = 0
    previous_face_coordinates = None
    is_pausing = False

    # 面部关键点的索引，选择眼睛、鼻子和嘴巴的关键点
    face_indices = list(range(36, 42)) + list(range(42, 48)) + [27]  # 鼻子、眼睛、嘴巴

    for i, frame_data in enumerate(landmarks_data):
        # 获取当前帧的面部坐标（x, y, z）
        current_face_coordinates = calculate_face_coordinates(frame_data['landmarks'], face_indices)

        # 如果是第一次计算面部坐标，初始化
        if previous_face_coordinates is None:
            previous_face_coordinates = current_face_coordinates
            continue

        # 计算坐标的变化量（x, y, z的变化量）
        delta_x = abs(current_face_coordinates[0] - previous_face_coordinates[0])
        delta_y = abs(current_face_coordinates[1] - previous_face_coordinates[1])
        delta_z = abs(current_face_coordinates[2] - previous_face_coordinates[2])

        # 判断坐标变化是否小于阈值（即进入停顿状态）
        if delta_x < threshold and delta_y < threshold and delta_z < threshold:
            if not is_pausing:
                is_pausing = True
                pause_duration = 1  # 新的停顿开始，计时
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)  # 记录停顿点的起始位置
            is_pausing = False
            pause_duration = 0

        previous_face_coordinates = current_face_coordinates

    # 处理最后可能的停顿结束
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)

    return pause_points





def generate_video_with_pose_data(input_video_path, output_video_path, output_json_path, pause_points=None):
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

                    all_frames_data.append({
                        "frame_number": frame_number,
                        "landmarks": aligned_landmarks
                    })

                    # 标注关键帧
                    if frame_number in pause_points:
                        # 在左上角添加文字“key frame”
                        cv2.putText(processed_image, 'Key Frame', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

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


# 停顿点检测并保存到 CSV 文件
def save_keyframe_data_to_csv(pause_points, all_frames_data, output_csv_path):
    """
    将关键帧的数据保存到 CSV 文件，包括帧数和对应的关键点数据
    :param pause_points: 停顿点的索引列表
    :param all_frames_data: 所有帧的骨架数据
    :param output_csv_path: 输出 CSV 文件路径
    """
    keyframe_data = []

    # 遍历每个停顿点
    for frame_number in pause_points:
        # 获取该帧的骨架数据
        frame_data = next((item for item in all_frames_data if item['frame_number'] == frame_number), None)

        if frame_data:
            landmarks = frame_data['landmarks']
            keyframe_points = {
                "frame_number": frame_number,
                "landmarks": [{"id": lm['id'], "x": lm['x'], "y": lm['y'], "z": lm['z'], "visibility": lm['visibility']}
                              for lm in landmarks]
            }
            keyframe_data.append(keyframe_points)

    # 将关键帧数据保存为 CSV 文件
    df = pd.DataFrame(keyframe_data)
    df.to_csv(output_csv_path, index=False)
    print(f"关键帧数据已成功保存到: {output_csv_path}")








# 使用示例：检测停顿点并保存关键帧数据到 CSV 文件
REFERENCE_JSON_PATH = "pose_data.json"  # 替换为摄像头视频的骨架数据文件路径
OUTPUT_CSV_PATH = "keyframe_data.csv"  # 输出关键帧数据的 CSV 文件路径

# 加载参考视频的骨架数据
with open(REFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
    reference_data = json.load(f)

# 检测停顿点
pause_points = detect_pause_points(reference_data, foot='right', threshold=0.01, min_pause_duration=5)

# 保存所有帧的骨架数据
all_frames_data = []
for frame_number, frame_data in enumerate(reference_data):
    all_frames_data.append({
        "frame_number": frame_number,
        "landmarks": frame_data['landmarks']
    })

# 保存关键帧数据到 CSV 文件
save_keyframe_data_to_csv(pause_points, all_frames_data, OUTPUT_CSV_PATH)







# 假设您希望同时根据面部、脚尖、膝盖、肩膀来检测停顿点
pause_points_foot = detect_pause_points(reference_data, foot='right', threshold=0.01, min_pause_duration=5)
pause_points_hip_left = detect_hip_pause_points(reference_data, side='left', threshold=0.01, min_pause_duration=5)
pause_points_knee_left = detect_knee_pause_points(reference_data, side='left', threshold=0.01, min_pause_duration=5)
pause_points_knee_right = detect_knee_pause_points(reference_data, side='right', threshold=0.01, min_pause_duration=5)
pause_points_shoulder_left = detect_shoulder_pause_points(reference_data, side='left', threshold=0.01, min_pause_duration=5)
pause_points_shoulder_right = detect_shoulder_pause_points(reference_data, side='right', threshold=0.01, min_pause_duration=5)
pause_points_face = detect_face_pause_points(reference_data, threshold=0.01, min_pause_duration=5)

# 将所有的停顿点合并
pause_points = sorted(set(pause_points_foot + pause_points_hip_left + pause_points_knee_left +
                          pause_points_knee_right + pause_points_shoulder_left + pause_points_shoulder_right +
                          pause_points_face))

# 生成并保存带有关键帧标注的视频
INPUT_VIDEO_PATH = "camera_video.mp4"  # 替换为输入的视频路径
OUTPUT_VIDEO_PATH = "output_with_keyframes.mp4"  # 输出带有关键帧标注的视频路径
OUTPUT_JSON_PATH = "processed_pose_data.json"  # 输出 JSON 文件路径

generate_video_with_pose_data(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_JSON_PATH, pause_points=pause_points)
