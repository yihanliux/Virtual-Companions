import json


def load_and_organize_pose_data(json_path):
    """
    读取一个 JSON 文件中的姿态数据，并将其整理为132个数组，
    每个数组对应33个关键点中某一个的坐标（x, y, z, visibility）。

    参数:
        json_path (str): 输入JSON文件的路径。

    返回:
        dict: 一个字典，键是像 'landmark_0_x' 这样的字符串，值是该数据点在所有帧中的数据列表。
              如果文件无法读取或为空，则返回一个空字典。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            all_frames_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 在'{json_path}'未找到文件")
        return {}
    except json.JSONDecodeError:
        print(f"错误: '{json_path}'处的文件不是一个有效的JSON文件。")
        return {}
    except Exception as e:
        print(f"发生意外错误: {e}")
        return {}

    if not all_frames_data:
        print("警告: JSON文件为空。")
        return {}

    # --- 1. 确定总帧数 ---
    # 我们通过找到最大的帧编号来确定数组的长度。
    # 这样做可以处理视频末尾可能存在没有检测到姿态的帧的情况。
    total_frames = 0
    if all_frames_data:
        total_frames = max(frame['frame_number'] for frame in all_frames_data) + 1

    print(f"检测到数据直至第 {total_frames - 1} 帧。将创建长度为 {total_frames} 的数组。")

    # --- 2. 初始化数据结构 ---
    # 我们创建一个字典来存放132个列表。
    # 列表预先用 `None` 填充，以应对那些没有检测到姿态的帧。
    organized_data = {}
    landmark_ids = range(33)  # 33个关键点
    attributes = ['x', 'y', 'z', 'visibility']  # 每个关键点有4个数据属性

    for i in landmark_ids:
        for attr in attributes:
            key_name = f"landmark_{i}_{attr}"
            organized_data[key_name] = [None] * total_frames

    # --- 3. 用JSON文件中的数据填充数组 ---
    print("正在整理数据...")
    for frame_data in all_frames_data:
        frame_number = frame_data['frame_number']
        landmarks = frame_data['landmarks']

        for landmark in landmarks:
            landmark_id = landmark['id']

            # 构建键名以找到正确的列表
            key_x = f"landmark_{landmark_id}_x"
            key_y = f"landmark_{landmark_id}_y"
            key_z = f"landmark_{landmark_id}_z"
            key_vis = f"landmark_{landmark_id}_visibility"

            # 将数据放入正确的列表中的正确位置（即帧编号对应的索引）
            organized_data[key_x][frame_number] = landmark['x']
            organized_data[key_y][frame_number] = landmark['y']
            organized_data[key_z][frame_number] = landmark['z']
            organized_data[key_vis][frame_number] = landmark['visibility']

    print("数据整理完成。")
    return organized_data