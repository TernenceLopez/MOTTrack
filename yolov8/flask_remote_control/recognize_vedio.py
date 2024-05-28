import time
from flask import Flask, request, jsonify, send_from_directory, send_file, url_for
from flask_cors import CORS
import os
import base64
from io import BytesIO
from PIL import Image
import shutil
import subprocess
import uuid
import json
from werkzeug.utils import secure_filename

# 不加这个会有CORS错误
# CORS（跨域资源共享）错误通常发生在前端尝试从与其不同的域、协议或端口访问资源时。
# 如果你的前端在不同的域名或端口上运行（例如，前端在http://localhost:3000而后端在http://localhost:5000），则需要在后端设置CORS以允许跨域请求。
app = Flask(__name__)
CORS(app)  # 允许所有域名的请求

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']

    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if video:
        # 保存视频文件到上传文件夹
        filename = secure_filename(video.filename)
        file_ext = os.path.splitext(filename)[1]
        unique_filename = str(uuid.uuid4()) + file_ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        video.save(filepath)

        processed_filepath = process_video(filepath)

        if processed_filepath:
            video_url = url_for('uploaded_file', filename=os.path.basename(processed_filepath), _external=True)
            check_status_url = url_for('check_status', filename=os.path.basename(processed_filepath), _external=True)
            return jsonify({'videoUrl': video_url, 'statusUrl': check_status_url})

    return jsonify({'error': '视频处理失败'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/check_status/<filename>')
def check_status(filename):
    # 简单模拟状态检查
    return jsonify({'status': 'complete'})


def process_video(input_path):
    try:
        output_filename = 'processed_' + os.path.basename(input_path)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        # 调用YOLOv8的detect.py脚本进行视频处理
        script_path = '/home/elvis/shareFolder/MOTTrack_Git/yolov8/ultralytics/yolo/v8/detect/predict.py'
        command = ['python', script_path, '--source', input_path, '--upload_dir', os.getcwd()]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 1:
            # 假设处理后的视频保存在output_path
            predict_result_path = os.path.join(
                                    os.path.dirname(
                                        os.path.dirname(
                                            os.path.dirname(os.path.abspath(__file__)))), 'runs/detect')
            # 使用最近修改时间获取视频的预测结果存放目录
            # # 获取当前目录下的所有文件夹
            # folders = [folder for folder in os.listdir(predict_result_path) if
            #            os.path.isdir(os.path.join(predict_result_path, folder))]
            # # 排序文件夹列表，按照最后修改时间降序排列
            # folders.sort(key=lambda x: os.path.getmtime(os.path.join(predict_result_path, x)), reverse=True)
            # predict_result_path = os.path.join(predict_result_path, folders[0], input_path.split('/')[-1])

            # 使用predict.py脚本中记录的映射关系获取视频预测结果存放目录
            with open(os.path.join(predict_result_path, 'data.json'), 'r') as file:
                json_data_from_file = file.read()
            # 反序列化 JSON 字符串回字典
            data_from_file = json.loads(json_data_from_file)
            predict_result_path = os.path.join(predict_result_path, data_from_file[input_path.split("/")[-1]], input_path.split("/")[-1])

            return predict_result_path
        else:
            print(f"Error processing video: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception during video processing: {e}")
        return None


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    app.run(debug=True)
