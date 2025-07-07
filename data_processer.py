import read_pose_data



if __name__ == "__main__":
    # 由你的第一个脚本所创建的 JSON 文件的路径
    INPUT_JSON_PATH = "pose_data.json"

    # 加载并处理数据
    pose_arrays = read_pose_data.load_and_organize_pose_data(INPUT_JSON_PATH)

    # --- 验证和使用示例 ---
    if pose_arrays:
        # 1. 检查创建的数组总数
        print(f"\n成功创建了 {len(pose_arrays)} 个数组。")  # 应该是 132

        # 2. 检查一个示例数组的长度（例如，鼻子的'x'坐标）
        # 鼻子是0号关键点
        nose_x_data = pose_arrays['landmark_0_x']
        print(f"'nose_x' 数组的长度: {len(nose_x_data)}")

        # 3. 打印鼻子 x 和 y 坐标的前10个值
        # 这有助于验证数据看起来是否正确。
        # 对于没有检测到姿态的帧，将打印 `None`。
        print("\n鼻子 (关键点 0) X坐标的前10个值:")
        # 使用列表推导进行格式化输出，保留4位小数
        print([round(val, 4) if val is not None else None for val in pose_arrays['landmark_0_x'][:10]])

        print("\n鼻子 (关键点 0) Y坐标的前10个值:")
        print([round(val, 4) if val is not None else None for val in pose_arrays['landmark_0_y'][:10]])

        # 4. 现在你可以轻松访问你想要的任何数据点。
        # 例如，获取第50帧时左手腕（关键点15）的可见性：
        try:
            left_wrist_visibility_frame_50 = pose_arrays.get('landmark_15_visibility', [])[50]
            print(f"\n在第50帧时左手腕 (关键点 15) 的可见性: {left_wrist_visibility_frame_50}")
        except IndexError:
            print("\n视频总帧数小于51帧，无法获取第50帧的数据。")