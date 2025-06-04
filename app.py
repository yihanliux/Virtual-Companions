from flask import Flask, render_template, request, jsonify
import os
import cv2
import mediapipe as mp
from datetime import datetime

# 初始化 Flask 应用
app = Flask(__name__)

# 上传文件保存路径（原图）
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 处理后图像输出路径（画完骨架图）
RESULTS_FOLDER = "static/results"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

# 固定的处理结果图片路径（可优化为动态命名）
RESULT_FILE = 'static/results/processed.jpg'

# 初始化 MediaPipe 的 Pose 模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(static_image_mode=True)

# 处理图像：识别姿态并绘制骨架
def process_pose_image(image_path: str, save_path: str):
    image = cv2.imread(image_path)
    if image is None:
        return False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite(save_path, image)
    return True

# 首页路由：显示上传页面
@app.route('/')
def index():
    return render_template('index.html')

# 上传接口：接收前端照片、处理并返回结果图路径
@app.route('/upload', methods=['POST'])
def upload():
    # 检查是否有文件字段
    if 'photo' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400

    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    # 保存上传的原图
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    filename = f"upload_{timestamp}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    photo.save(file_path)

    # 处理图像并保存处理后图
    success = process_pose_image(file_path, RESULT_FILE)
    if not success:
        return jsonify({'status': 'error', 'message': 'Image processing failed'}), 500

    # 返回处理图像的访问路径
    return jsonify({'status': 'success', 'image_url': '/' + RESULT_FILE})

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
