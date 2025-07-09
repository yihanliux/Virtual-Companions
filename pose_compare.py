import math
import read_pose_data
import matplotlib.pyplot as plt
import json

# ========== 帧数对齐函数 ==========
def align_frame_count(data_dict, target_frame_count):
    aligned = {}
    for key, series in data_dict.items():
        if not isinstance(series, list):
            aligned[key] = series
            continue
        current_len = len(series)
        if current_len == target_frame_count:
            aligned[key] = series
        elif current_len < target_frame_count:
            aligned[key] = series + [None] * (target_frame_count - current_len)
        else:
            aligned[key] = series[:target_frame_count]
    return aligned

# 步骤 1：计算身高缩放比例（默认鼻子到左脚踝）
def compute_scale_factors_xy(data, top_id=0, bottom_id=27, left_id=11, right_id=12):
    """
    同时计算 y 和 x 方向的缩放比例。
    y_ratio 基于鼻子到脚踝，x_ratio 基于左右肩宽。
    返回 [(x_ratio1, y_ratio1), (x_ratio2, y_ratio2), ...]
    """
    y_top = data[f"landmark_{top_id}_y"]
    y_bottom = data[f"landmark_{bottom_id}_y"]
    x_left = data[f"landmark_{left_id}_x"]
    x_right = data[f"landmark_{right_id}_x"]
    result = []
    for yt, yb, xl, xr in zip(y_top, y_bottom, x_left, x_right):
        if None in (yt, yb, xl, xr):
            result.append((None, None))
        else:
            h = abs(yb - yt)
            w = abs(xr - xl)
            y_ratio = 1.0 / h if h > 0 else None
            x_ratio = 1.0 / w if w > 0 else None
            result.append((x_ratio, y_ratio))
    return result


def apply_scale_xy(data, scale_pairs):
    """
    对 x 和 y 使用不同的缩放比例；z 保持与 y 同步；其他不变
    """
    scaled = {}
    for k, v in data.items():
        if k.endswith("_x"):
            scaled[k] = [val * s[0] if val is not None and s[0] is not None else None for val, s in zip(v, scale_pairs)]
        elif k.endswith("_y") or k.endswith("_z"):
            scaled[k] = [val * s[1] if val is not None and s[1] is not None else None for val, s in zip(v, scale_pairs)]
        else:
            scaled[k] = v
    return scaled

# 步骤 3：将肩膀中心平移为原点（便于对比姿态）
def apply_centering(data):
    l_shoulder_x = data["landmark_11_x"]
    r_shoulder_x = data["landmark_12_x"]
    l_shoulder_y = data["landmark_11_y"]
    r_shoulder_y = data["landmark_12_y"]
    cx = [(lx + rx) / 2 if lx is not None and rx is not None else None for lx, rx in zip(l_shoulder_x, r_shoulder_x)]
    cy = [(ly + ry) / 2 if ly is not None and ry is not None else None for ly, ry in zip(l_shoulder_y, r_shoulder_y)]
    centered = {}
    for k, v in data.items():
        if k.endswith("_x"):
            centered[k] = [val - c if val is not None and c is not None else None for val, c in zip(v, cx)]
        elif k.endswith("_y"):
            centered[k] = [val - c if val is not None and c is not None else None for val, c in zip(v, cy)]
        else:
            centered[k] = v
    return centered

# 获取某一帧某个关键点的 (x, y, z)
def extract_point(data, landmark_id, frame_idx):
    x = data[f"landmark_{landmark_id}_x"][frame_idx]
    y = data[f"landmark_{landmark_id}_y"][frame_idx]
    z = data[f"landmark_{landmark_id}_z"][frame_idx]
    return (x, y, z) if x is not None and y is not None and z is not None else None

# 计算两向量夹角（单位：度）
def compute_angle_between_vectors(p1, p2, q1, q2):
    v1 = [b - a for a, b in zip(p1, p2)]
    v2 = [b - a for a, b in zip(q1, q2)]
    dot = sum(a*b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a*a for a in v1))
    norm2 = math.sqrt(sum(a*a for a in v2))
    if norm1 == 0 or norm2 == 0:
        return None
    cos_theta = dot / (norm1 * norm2)
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return math.acos(cos_theta) * 180 / math.pi

# 提取标准者自身相邻帧之间的动作角度变化
def extract_angle_sequence(data, joint_from=11, joint_to=23):
    angles = []
    for i in range(1, len(data["landmark_0_x"])):
        p1 = extract_point(data, joint_from, i - 1)
        p2 = extract_point(data, joint_to, i - 1)
        p3 = extract_point(data, joint_from, i)
        p4 = extract_point(data, joint_to, i)
        if None not in (p1, p2, p3, p4):
            angle = compute_angle_between_vectors(p1, p2, p3, p4)
        else:
            angle = None
        angles.append(angle)
    return angles

# 分段对比
def compare_segments(std_data, sub_data, segment_list, keyframes=None):
    results = {}
    total_frames = len(std_data['landmark_0_x'])
    frames = keyframes if keyframes is not None else range(total_frames)
    for name, a, b in segment_list:
        series = []
        for i in frames:
            p1 = extract_point(std_data, a, i)
            p2 = extract_point(std_data, b, i)
            q1 = extract_point(sub_data, a, i)
            q2 = extract_point(sub_data, b, i)
            if None not in (p1, p2, q1, q2):
                angle = compute_angle_between_vectors(p1, p2, q1, q2)
            else:
                angle = None
            series.append(angle)
        avg = (sum(a for a in series if a is not None) / sum(1 for a in series if a is not None)) if any(series) else None
        results[name] = {'series': series, 'avg': avg}
    return results

# 计算整体平均角度误差
def compute_overall_average(results):
    valid_avgs = [info['avg'] for info in results.values() if info['avg'] is not None]
    return sum(valid_avgs) / len(valid_avgs) if valid_avgs else None

# 可视化误差柱状图
def bar_plot(avg_dict):
    names = list(avg_dict.keys())
    values = [avg_dict[k] if avg_dict[k] is not None else 0 for k in names]
    plt.figure(figsize=(12, 5))
    plt.bar(names, values, color='salmon')
    plt.ylabel('Average Angle Error (°)')
    plt.title('Pose Imitation Error by Segment (Keyframes Only)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ========== 停顿检测辅助函数 ==========

# Extract coordinates from landmarks
def calculate_coordinate(landmarks, idx):
    lm = landmarks[idx]
    return (lm['x'], lm['y'], lm['z'])

# Detect pause points with general body landmarks
def detect_pause_points_general(landmarks_data, idx, threshold=0.01, min_pause_duration=5):
    pause_points = []
    pause_duration = 0
    previous = None
    is_pausing = False
    for i, frame_data in enumerate(landmarks_data):
        current = calculate_coordinate(frame_data['landmarks'], idx)
        if previous is None:
            previous = current
            continue
        delta = [abs(a - b) for a, b in zip(current, previous)]
        if all(d < threshold for d in delta):
            if not is_pausing:
                is_pausing = True
                pause_duration = 1
            else:
                pause_duration += 1
        else:
            if is_pausing and pause_duration >= min_pause_duration:
                pause_points.append(i - pause_duration)
            is_pausing = False
            pause_duration = 0
        previous = current
    if is_pausing and pause_duration >= min_pause_duration:
        pause_points.append(len(landmarks_data) - pause_duration)
    return pause_points

# ========== 融合关键帧提取函数 ==========

def extract_keyframes_combined_with_pauses(std_data, sub_data, ref_json_path):
    from math import sqrt
    def auto_threshold_extract(series):
        diffs = []
        for i in range(1, len(series)):
            if series[i] is None or series[i - 1] is None:
                diffs.append(0)
            else:
                diffs.append(abs(series[i] - series[i - 1]))
        if not diffs:
            return [0]
        mean_diff = sum(diffs) / len(diffs)
        std_diff = sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
        return [i + 1 for i, d in enumerate(diffs) if d > (mean_diff + std_diff)]

    std_change = extract_angle_sequence(std_data)
    imitation_error_series = compare_segments(std_data, sub_data, [("ref", 11, 23)])['ref']['series'][1:]

    keys_std = set(auto_threshold_extract(std_change))
    keys_err = set(auto_threshold_extract(imitation_error_series))
    keys_dual = keys_std | keys_err

    with open(ref_json_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    pause_foot = set(detect_pause_points_general(ref_data, 32))
    pause_hip = set(detect_pause_points_general(ref_data, 23))
    pause_knee = set(detect_pause_points_general(ref_data, 27))
    pause_shldr = set(detect_pause_points_general(ref_data, 11))


    pause_all = pause_foot & pause_hip & pause_knee & pause_shldr


    return sorted(keys_dual & pause_all)

# 主函数：完整流程
def main():
    standard_path = "pose_data_7s.json"
    subject_path = "pose_data_zyx.json"
    std_raw = read_pose_data.load_and_organize_pose_data(standard_path)
    sub_raw = read_pose_data.load_and_organize_pose_data(subject_path)

    TARGET_FRAMES = 211
    std_raw = align_frame_count(std_raw, TARGET_FRAMES)
    sub_raw = align_frame_count(sub_raw, TARGET_FRAMES)

    # 使用 XY 方向缩放 + 中心对齐
    scale_pairs_std = compute_scale_factors_xy(std_raw)
    scale_pairs_sub = compute_scale_factors_xy(sub_raw)

    std = apply_centering(apply_scale_xy(std_raw, scale_pairs_std))
    sub = apply_centering(apply_scale_xy(sub_raw, scale_pairs_sub))

    segment_list = []
    excluded = set(range(0, 11)) | set(range(17, 23)) | {29, 30, 31, 32}  # Exclude finger points
    total_points = 33
    for i in range(total_points):
        for j in range(i + 1, total_points):
            if i in excluded or j in excluded:
                continue
            if {i, j} <= {29, 30, 31, 32}:  # Exclude pairwise toe point comparisons
                continue
            segment_list.append((f"{i}-{j}", i, j))

    keyframes = extract_keyframes_combined_with_pauses(std, sub, ref_json_path=standard_path)
    results = compare_segments(std, sub, segment_list, keyframes=keyframes)
    overall_avg = compute_overall_average(results)

    for name, info in sorted(results.items()):
        avg = info['avg']
        if avg is not None:
            print(f"{name:<8}: 平均角度差 = {avg:.2f}°")
        else:
            print(f"{name:<8}: 平均角度差 = N/A")

    if overall_avg is not None:
        print(f"\n总体平均角度差 (关键帧): {overall_avg:.2f}°")
    else:
        print("\n总体平均角度差 (关键帧): N/A")

    bar_plot({k: v['avg'] for k, v in results.items()})

if __name__ == "__main__":
    main()