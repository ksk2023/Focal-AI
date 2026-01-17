"""
AI Photography Director - 姿势录制工具
用于录制和保存自定义姿势到 poses.json
"""

import cv2
import json
import numpy as np
import mediapipe as mp
from datetime import datetime


def record_pose():
    """
    录制姿势工具
    - 按 's' 保存当前姿势
    - 按 'q' 退出
    """
    
    # 初始化 MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("=" * 50)
    print("📸 姿势录制工具")
    print("=" * 50)
    print("操作说明：")
    print("  [s] - 保存当前姿势")
    print("  [q] - 退出程序")
    print("=" * 50)
    
    # 加载现有 poses.json
    poses_file = "poses.json"
    try:
        with open(poses_file, 'r', encoding='utf-8') as f:
            poses_data = json.load(f)
    except FileNotFoundError:
        poses_data = {}
    
    current_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 翻转图像（镜像）
        frame = cv2.flip(frame, 1)
        
        # 转换颜色
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测姿态
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # 绘制骨骼
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            # 保存当前关键点
            current_landmarks = [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
            
            # 显示状态
            cv2.putText(frame, "Pose Detected! Press 's' to save", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No pose detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示已保存的姿势数量
        cv2.putText(frame, f"Saved poses: {len(poses_data)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Pose Recorder", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s') and current_landmarks:
            # 保存姿势
            pose_name = input("\n请输入姿势名称(英文，如 my_pose): ").strip()
            pose_desc = input("请输入姿势描述(中文): ").strip()
            
            if pose_name:
                poses_data[pose_name] = {
                    "name": pose_desc or pose_name,
                    "description": pose_desc or f"自定义姿势 - {pose_name}",
                    "landmarks": current_landmarks
                }
                
                # 保存到文件
                with open(poses_file, 'w', encoding='utf-8') as f:
                    json.dump(poses_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ 已保存姿势: {pose_name}")
                print(f"   共 {len(poses_data)} 个姿势")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 录制完成！")


if __name__ == "__main__":
    record_pose()
