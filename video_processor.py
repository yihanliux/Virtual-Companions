import cv2
import mediapipe as mp
import json
from tqdm import tqdm

# --- 初始化 MediaPipe Pose ---
# 初始化 MediaPipe solutions，以便我们使用姿态检测功能
mp_pose = mp.solutions.pose
# 初始化绘图工具，用于在图像上绘制关键点和连接线
mp_drawing = mp.solutions.drawing_utils
# 创建 Pose 实例
# - static_image_mode=False:  处理视频流，而不是单张静态图片。
# - model_complexity=2:       使用最高精度的模型，效果最好但速度最慢 (可选 0, 1, 2)。
# - smooth_landmarks=True:    平滑关键点，减少抖动。
# - min_detection_confidence=0.5: 只有当检测置信度高于 50% 时，才认为检测成功。
# - min_tracking_confidence=0.5:  只有当跟踪置信度高于 50% 时，才继续跟踪。
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def process_frame_and_extract_landmarks(image):
    """
    处理单帧图像，进行姿态检测、绘制关键点并提取结构化数据。

    Args:
        image (numpy.ndarray): 输入的单帧图像 (BGR 格式)。

    Returns:
        tuple:
            - numpy.ndarray: 绘制了姿态关键点的图像。
            - list or None: 包含所有关键点结构化数据的列表。如果未检测到姿态，则返回 None。
    """
    # 将图像从 BGR 格式转换为 RGB 格式，因为 MediaPipe 需要 RGB 图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 进行姿态检测
    results = pose.process(image_rgb)

    # 检查是否成功检测到姿态关键点
    if not results.pose_landmarks:
        # 如果没有检测到，则返回原始图像和 None
        return image, None

    # --- 提取关键点数据 ---
    # MediaPipe 返回的 `pose_landmarks` 包含了 33 个身体关键点
    # 每个关键点包含 x, y, z 和 visibility
    # x, y, z 是归一化坐标 (0.0 到 1.0)，z 代表深度
    # visibility 代表该点是否可见
    landmarks_data = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        landmarks_data.append({
            'id': i,
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility
        })

    # --- 在图像上绘制关键点和连接线 ---
    # 我们在原始 BGR 图像上进行绘制，以便可以直接用于视频输出
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )

    return image, landmarks_data


def generate_video_with_pose_data(input_video_path, output_video_path, output_json_path):
    """
    读取视频，处理每一帧，保存为带有关节点的新视频，并同时将姿态数据保存为 JSON 文件。

    Args:
        input_video_path (str): 输入视频文件的路径。
        output_video_path (str): 处理后输出视频的保存路径。
        output_json_path (str): 姿态数据 JSON 文件的保存路径。
    """
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"错误: 无法打开视频文件 '{input_video_path}'")
    except Exception as e:
        print(e)
        return

    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 'mp4v' 编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    all_frames_data = []  # 用于存储所有帧的数据

    print("开始处理视频...")
    # 使用 tqdm 创建进度条
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        frame_number = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            try:
                # 处理当前帧，获取带有关节点的图像和关键点数据
                processed_image, landmarks = process_frame_and_extract_landmarks(frame)

                # 将处理后的图像写入新视频文件
                out.write(processed_image)

                # 如果成功提取到关键点数据，则记录下来
                if landmarks:
                    all_frames_data.append({
                        "frame_number": frame_number,
                        "landmarks": landmarks
                    })

            except Exception as e:
                print(f"处理第 {frame_number} 帧时出错: {e}")
                # 即使处理失败，也写入原始帧以保持视频连续性
                out.write(frame)

            pbar.update(1)
            frame_number += 1

    print("视频处理完成，正在保存数据...")

    # --- 保存数据到 JSON 文件 ---
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # 使用 json.dump 将数据写入文件，indent=4 使其格式化，更易读
            json.dump(all_frames_data, f, ensure_ascii=False, indent=4)
        print(f"姿态数据成功保存到: {output_json_path}")
    except Exception as e:
        print(f"保存 JSON 文件时出错: {e}")

    # 释放所有资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理后的视频成功保存到: {output_video_path}")


if __name__ == "__main__":
    # --- 配置输入和输出文件路径 ---
    INPUT_VIDEO_PATH = "video.mp4"  # 替换为你的输入视频路径
    OUTPUT_VIDEO_PATH = "output_video.mp4"  # 这是处理后带有关节点的视频
    OUTPUT_JSON_PATH = "pose_data.json"  # 这是提取出的姿态数据文件

    generate_video_with_pose_data(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_JSON_PATH)