import os
import face_recognition
import numpy as np
from PIL import Image
import shutil

def classify_faces(source_dir, dest_dir, tolerance=0.4):
    known_encodings = []
    known_labels = []
    current_label = 0

    for filename in os.listdir(source_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        filepath = os.path.join(source_dir, filename)
        try:
            # 加载图片并优化处理速度
            image = Image.open(filepath)
            image.thumbnail((500, 500))  # 缩小图片尺寸
            image_np = np.array(image)

            # 提取人脸编码
            face_encodings = face_recognition.face_encodings(image_np)
            if not face_encodings:
                print(f"未检测到人脸: {filename}")
                continue
            face_encoding = face_encodings[0]

            # 比较已知编码
            if known_encodings:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance = np.min(face_distances)
                min_index = np.argmin(face_distances)
                if min_distance <= tolerance:
                    label = known_labels[min_index]
                else:
                    current_label += 1
                    known_encodings.append(face_encoding)
                    known_labels.append(current_label)
                    label = current_label
            else:
                # 第一张图片
                current_label = 1
                known_encodings.append(face_encoding)
                known_labels.append(current_label)
                label = current_label

            # 创建目标文件夹并复制文件
            label_dir = os.path.join(dest_dir, f"person_{label}")
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(filepath, os.path.join(label_dir, filename))
            print(f"已处理: {filename} → person_{label}")

        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

# 使用示例
classify_faces("photos", "persons")