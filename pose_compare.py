import math
import read_pose_data
import matplotlib.pyplot as plt

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
def compute_scale_factors(data, top_id=0, bottom_id=27):
    y_top = data[f"landmark_{top_id}_y"]
    y_bottom = data[f"landmark_{bottom_id}_y"]
    scales = []
    for yt, yb in zip(y_top, y_bottom):
        if yt is not None and yb is not None:
            h = abs(yb - yt)
            scales.append(1.0 / h if h > 0 else None)
        else:
            scales.append(None)
    return scales

# 步骤 2：将缩放比例应用于所有 x/y/z 坐标
def apply_scale(data, scales):
    scaled = {}
    for k, v in data.items():
        if any(k.endswith(s) for s in ['_x', '_y', '_z']):
            scaled[k] = [val * s if val is not None and s is not None else None for val, s in zip(v, scales)]
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

# 关键帧提取（动作变化 + 模仿误差变化）
def extract_keyframes_dual_criteria(std_data, sub_data, joint_from=11, joint_to=23):
    from math import sqrt
    std_change = extract_angle_sequence(std_data, joint_from, joint_to)
    imitation_error_series = compare_segments(std_data, sub_data, [("ref", joint_from, joint_to)])['ref']['series'][1:]
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
    keys_std = auto_threshold_extract(std_change)
    keys_err = auto_threshold_extract(imitation_error_series)
    return sorted(set(keys_std + keys_err))

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

# 主函数：完整流程
def main():
    standard_path = "pose_data_7s.json"
    subject_path = "pose_data_zyx.json"
    std_raw = read_pose_data.load_and_organize_pose_data(standard_path)
    sub_raw = read_pose_data.load_and_organize_pose_data(subject_path)

    # 统一帧数为 211 帧
    TARGET_FRAMES = 211
    std_raw = align_frame_count(std_raw, TARGET_FRAMES)
    sub_raw = align_frame_count(sub_raw, TARGET_FRAMES)

    std = apply_centering(apply_scale(std_raw, compute_scale_factors(std_raw)))
    sub = apply_centering(apply_scale(sub_raw, compute_scale_factors(sub_raw)))

    segment_list = []
    excluded = set(range(0, 11))
    total_points = 33
    for i in range(total_points):
        for j in range(i+1, total_points):
            if i not in excluded and j not in excluded:
                segment_list.append((f"{i}-{j}", i, j))

    keyframes = extract_keyframes_dual_criteria(std, sub, joint_from=11, joint_to=23)
    results = compare_segments(std, sub, segment_list, keyframes=keyframes)
    overall_avg = compute_overall_average(results)

    for name, info in sorted(results.items()):
        print(f"{name:<8}: 平均角度差 = {info['avg']:.2f}°")
    print(f"\n总体平均角度差 (关键帧): {overall_avg:.2f}°")

    bar_plot({k: v['avg'] for k, v in results.items()})

if __name__ == "__main__":
    main()