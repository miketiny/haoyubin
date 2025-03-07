import os
import face_recognition
import numpy as np
from PIL import Image
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import time

class FaceClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人脸照片自动分类工具")
        self.running = False  # 运行状态标志
        self.queue = queue.Queue()  # 用于线程间通信

        # 界面布局
        self.setup_ui()
        
        # 启动队列监听
        self.root.after(100, self.process_queue)

    def setup_ui(self):
        self.frame = ttk.Frame(self.root, padding="20")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.progress_label = ttk.Label(self.frame, text="0/0")
        self.progress_label.grid(row=2, column=3, padx=5)  # 放在进度条右侧
        
        # 输入目录
        ttk.Label(self.frame, text="选择照片文件夹:").grid(row=0, column=0, sticky=tk.W)
        self.input_dir = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.input_dir, width=40).grid(row=0, column=1)
        ttk.Button(self.frame, text="浏览", command=self.select_input_dir).grid(row=0, column=2)
        
        # 输出目录
        ttk.Label(self.frame, text="选择输出文件夹:").grid(row=1, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar()
        ttk.Entry(self.frame, textvariable=self.output_dir, width=40).grid(row=1, column=1)
        ttk.Button(self.frame, text="浏览", command=self.select_output_dir).grid(row=1, column=2)
        
        # 进度条
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=3, pady=10, sticky=tk.EW)
        
        # 控制按钮区域
        self.btn_frame = ttk.Frame(self.frame)
        self.btn_frame.grid(row=3, column=0, columnspan=3, pady=10)
        self.start_btn = ttk.Button(self.btn_frame, text="开始分类", command=self.start_classification)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.cancel_btn = ttk.Button(self.btn_frame, text="取消", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def select_input_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
    
    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
    
    def start_classification(self):
        if self.running:
            return
        
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        
        if not input_dir or not output_dir:
            messagebox.showerror("错误", "请先选择输入和输出文件夹！")
            return
        
        try:
            # 初始化状态
            self.running = True
            self.start_btn.config(state=tk.DISABLED)
            self.cancel_btn.config(state=tk.NORMAL)
            self.progress['value'] = 0
            
            # 启动子线程
            self.worker_thread = threading.Thread(
                target=self.classify_faces,
                args=(input_dir, output_dir),
                daemon=True
            )
            self.worker_thread.start()
        
        except Exception as e:
            self.handle_error(f"无法启动线程: {str(e)}")
    
    def cancel_processing(self):
        self.running = False
        self.cancel_btn.config(state=tk.DISABLED)
        messagebox.showinfo("提示", "正在停止处理...")
    
    def process_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg['type'] == 'progress':
                    self.progress['value'] = msg['value']
                    self.progress_label.config(text=f"{msg['current']}/{msg['total']}")
                elif msg['type'] == 'error':
                    self.handle_error(msg['message'])
                elif msg['type'] == 'complete':
                    self.on_complete()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
    
    def handle_error(self, message):
        self.running = False
        self.reset_ui_state()
        messagebox.showerror("错误", message)
    
    def on_complete(self):
        self.running = False
        self.reset_ui_state()
        messagebox.showinfo("完成", "照片分类已完成！")
    
    def reset_ui_state(self):
        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
    
    def classify_faces(self, source_dir, dest_dir, tolerance=0.4):
        try:
            known_encodings = []
            known_labels = []
            current_label = 0
            
            files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_files = len(files)
            
            for idx, filename in enumerate(files, 1):
                if not self.running:
                    break
                
                filepath = os.path.join(source_dir, filename)
                try:
                    # 发送进度更新到主线程
                    progress = (idx / total_files) * 100
                    self.queue.put({'type': 'progress', 'value': progress, 'current': idx, 'total': total_files})
                    
                    # 处理单张图片
                    image = Image.open(filepath)
                    image.thumbnail((500, 500))
                    image_np = np.array(image)
                    
                    face_encodings = face_recognition.face_encodings(image_np)
                    if not face_encodings:
                        continue
                    face_encoding = face_encodings[0]
                    
                    # 匹配逻辑
                    if known_encodings:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        min_distance = np.min(face_distances)
                        min_index = np.argmin(face_distances)
                        label = known_labels[min_index] if min_distance <= tolerance else (current_label := current_label + 1)
                    else:
                        current_label = 1
                        label = current_label
                        min_distance = 1
                    
                    # 存储结果
                    if min_distance > tolerance:  # type: ignore
                        known_encodings.append(face_encoding)
                        known_labels.append(label)
                    
                    label_dir = os.path.join(dest_dir, f"person_{label}")
                    os.makedirs(label_dir, exist_ok=True)
                    shutil.copy(filepath, os.path.join(label_dir, filename))
                
                except Exception as e:
                    print(f"跳过文件 {filename}: {str(e)}")
            
            if self.running:
                self.queue.put({'type': 'complete'})
        
        except Exception as e:
            self.queue.put({'type': 'error', 'message': str(e)})

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceClassifierApp(root)
    root.mainloop()