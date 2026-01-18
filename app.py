"""
AI Photography Director - 完整版 Streamlit 前端应用
整合版：实时摄像头 + MediaPipe + 姿态匹配 + GPT分析 + 语音反馈 + 自动拍照
"""

import streamlit as st
import cv2
import numpy as np
import json
import time
import base64
import httpx
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# 导入姿态匹配模块
from pose_matcher import (
    Landmark,
    calculate_pose_similarity,
    get_feedback_instruction,
    get_detailed_analysis,
    TARGET_POSES,
)

# ==================== 页面配置 ====================

st.set_page_config(
    page_title="AI 拍照助手",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 加载 poses.json ====================

@st.cache_data
def load_poses_data():
    """加载预设姿势数据"""
    poses_file = Path(__file__).parent / "poses.json"
    try:
        with open(poses_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

POSES_DATA = load_poses_data()

# ==================== 自定义样式 ====================

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.7));
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .score-high { color: #00c853; text-shadow: 0 0 20px rgba(0,200,83,0.3); }
    .score-medium { color: #ff9800; text-shadow: 0 0 20px rgba(255,152,0,0.3); }
    .score-low { color: #f44336; text-shadow: 0 0 20px rgba(244,67,54,0.3); }
    
    .feedback-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        font-size: 1.1rem;
    }
    .analyzing-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    .perfect-banner {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        border-radius: 1rem;
        animation: celebratePulse 0.3s ease-in-out infinite alternate;
        box-shadow: 0 8px 32px rgba(0,200,83,0.4);
    }
    @keyframes celebratePulse {
        from { transform: scale(1); }
        to { transform: scale(1.05); }
    }
    .countdown {
        font-size: 8rem;
        font-weight: bold;
        text-align: center;
        color: #f64f59;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.2);
        animation: countdownPop 0.5s ease-out;
    }
    @keyframes countdownPop {
        0% { transform: scale(1.5); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #00c853 100%);
        border-radius: 10px;
    }
    .guide-step {
        background: white;
        border-radius: 1rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .guide-step.active {
        border-left: 4px solid #667eea;
        background: linear-gradient(90deg, #667eea10, white);
    }
    .guide-step.completed {
        border-left: 4px solid #00c853;
        background: linear-gradient(90deg, #00c85310, white);
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 MediaPipe ====================

# MediaPipe 0.10.x 版本兼容性处理
try:
    # 新版本 MediaPipe (0.10.14+) 需要使用 legacy 模块
    from mediapipe.python.solutions import pose as mp_pose_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    
    class MPPoseWrapper:
        """MediaPipe Pose 包装类"""
        POSE_CONNECTIONS = mp_pose_module.POSE_CONNECTIONS
        
        @staticmethod
        def Pose(**kwargs):
            return mp_pose_module.Pose(**kwargs)
    
    mp_pose = MPPoseWrapper()
    
except ImportError:
    try:
        # 旧版本 MediaPipe
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError:
        # 最新版本使用直接导入
        from mediapipe.tasks.python.vision import PoseLandmarker
        from mediapipe.tasks.python.vision import PoseLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions
        import cv2
        
        # 简化的绘制函数
        class SimpleMPPose:
            POSE_CONNECTIONS = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28)
            ]
            
            @staticmethod
            def Pose(**kwargs):
                return None
        
        mp_pose = SimpleMPPose()
        mp_drawing = None
        mp_drawing_styles = None

# ==================== Session State 初始化 ====================

def init_session_state():
    defaults = {
        "match_score": 0.0,
        "high_score_start": None,
        "captured_photos": [],
        "show_perfect": False,
        "feedback": "准备好后点击「开始引导」",
        "target_pose": "standing_casual",
        "workflow_stage": "idle",  # idle -> analyzing -> guiding -> countdown -> captured
        "ai_analysis": None,
        "countdown_value": None,
        "is_analyzing": False,
        "voice_enabled": True,
        "auto_capture_threshold": 85,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==================== API 调用函数 ====================

async def analyze_image_async(image_base64: str, user_message: str = None) -> Dict:
    """异步调用后端 API 分析图片"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:8000/analyze_image",
                json={
                    "image_base64": image_base64,
                    "user_message": user_message
                }
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_pose_landmarks_for_drawing(pose_id: str) -> List[tuple]:
    """从 poses.json 获取用于绘制的关键点坐标"""
    if pose_id not in POSES_DATA:
        return []
    return POSES_DATA[pose_id].get("landmarks", [])


# ==================== 语音反馈 (浏览器端 TTS) ====================

def speak_text_js(text: str):
    """使用浏览器内置 TTS 播放语音"""
    js_code = f"""
    <script>
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance("{text}");
            utterance.lang = 'zh-CN';
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            speechSynthesis.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0)


# ==================== 视频处理类 ====================

class PoseVideoProcessor(VideoProcessorBase):
    """实时视频处理：骨骼检测 + 目标姿势叠加 + 匹配度计算"""
    
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.current_score = 0.0
        self.current_feedback = ""
        self.frame_count = 0
        self.target_pose_id = "standing_casual"
        self.show_target_overlay = True
        self.last_frame = None
        
    def draw_target_pose_from_landmarks(self, image: np.ndarray, pose_id: str) -> np.ndarray:
        """从 poses.json 中的关键点绘制目标姿势"""
        landmarks = get_pose_landmarks_for_drawing(pose_id)
        if not landmarks or len(landmarks) < 17:
            return image
            
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # 转换为像素坐标
        points = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        
        # MediaPipe Pose 连接定义 (上半身关键连接)
        connections = [
            (11, 12),  # 双肩
            (11, 13), (13, 15),  # 左臂
            (12, 14), (14, 16),  # 右臂
            (11, 23), (12, 24),  # 肩到髋
            (23, 24),  # 双髋
        ]
        
        # 绘制虚线连接
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                self.draw_dashed_line(overlay, points[start_idx], points[end_idx], 
                                     (0, 255, 120), 3, 15)
        
        # 绘制关键点
        key_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # 肩、肘、腕、髋
        for idx in key_indices:
            if idx < len(points):
                cv2.circle(overlay, points[idx], 15, (0, 255, 120), 2)
                cv2.circle(overlay, points[idx], 8, (0, 255, 120), -1)
        
        # 混合叠加
        alpha = 0.35
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    def draw_dashed_line(self, img, pt1, pt2, color, thickness, dash_length):
        """绘制虚线"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist < 1:
            return
        dashes = max(int(dist / dash_length), 1)
        
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            start = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio),
                    int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
            end = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio),
                  int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
            cv2.line(img, start, end, color, thickness)
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """处理每一帧视频"""
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)  # 镜像
        
        self.last_frame = image.copy()
        
        # MediaPipe 姿态检测
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        # 根据工作流阶段决定是否显示目标姿势
        workflow_stage = st.session_state.get("workflow_stage", "idle")
        
        if workflow_stage in ["guiding", "countdown"] and self.show_target_overlay:
            target_pose = st.session_state.get("target_pose", self.target_pose_id)
            image = self.draw_target_pose_from_landmarks(image, target_pose)
        
        # 绘制用户骨骼 + 计算匹配度
        if results.pose_landmarks:
            # 白色骨骼线
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )
            
            # 计算匹配度 (每3帧一次)
            self.frame_count += 1
            if self.frame_count % 3 == 0 and workflow_stage == "guiding":
                try:
                    landmarks = [Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) 
                                for lm in results.pose_landmarks.landmark]
                    target_pose = st.session_state.get("target_pose", "standing_casual")
                    
                    self.current_score = calculate_pose_similarity(landmarks, target_pose)
                    feedback = get_feedback_instruction(self.current_score, landmarks, target_pose)
                    self.current_feedback = feedback or "完美！保持住！"
                    
                    st.session_state.match_score = self.current_score
                    st.session_state.feedback = self.current_feedback
                except:
                    pass
        
        # 绘制 UI 覆盖层
        self._draw_ui_overlay(image, workflow_stage)
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    def _draw_ui_overlay(self, image, stage):
        """绘制 UI 覆盖信息"""
        h, w = image.shape[:2]
        
        if stage == "analyzing":
            # 分析中状态
            cv2.rectangle(image, (w//4, h//2-40), (3*w//4, h//2+40), (0,0,0), -1)
            cv2.rectangle(image, (w//4, h//2-40), (3*w//4, h//2+40), (255,180,0), 2)
            cv2.putText(image, "Analyzing...", (w//2-80, h//2+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,180,0), 2)
        
        elif stage == "guiding":
            # 显示分数
            score = st.session_state.get("match_score", 0)
            color = (0,255,0) if score >= 70 else (0,165,255) if score >= 50 else (0,0,255)
            
            cv2.rectangle(image, (10, 10), (220, 70), (0,0,0), -1)
            cv2.rectangle(image, (10, 10), (220, 70), color, 2)
            cv2.putText(image, f"Match: {score:.0f}%", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        elif stage == "countdown":
            # 倒计时
            countdown = st.session_state.get("countdown_value", 3)
            if countdown:
                cv2.rectangle(image, (w//2-60, h//2-80), (w//2+60, h//2+80), (0,0,0), -1)
                cv2.putText(image, str(countdown), (w//2-30, h//2+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0,255,255), 4)


# ==================== 主应用界面 ====================

def main():
    # 标题
    st.markdown('<h1 class="main-header">📸 AI 拍照助手</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">✨ 让每一张照片都完美 · AI 智能引导拍摄</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 选择目标姿势
        pose_options = {k: f"{v.get('name', k)}" for k, v in POSES_DATA.items()}
        if not pose_options:
            pose_options = {"standing_casual": "自然站立"}
        
        selected_pose = st.selectbox(
            "🎭 目标姿势",
            options=list(pose_options.keys()),
            format_func=lambda x: pose_options.get(x, x)
        )
        st.session_state.target_pose = selected_pose
        
        if selected_pose in POSES_DATA:
            st.info(f"📝 {POSES_DATA[selected_pose].get('description', '')}")
        
        st.divider()
        
        # 设置
        st.session_state.auto_capture_threshold = st.slider(
            "🎯 自动拍照阈值", 70, 100, 85
        )
        st.session_state.voice_enabled = st.checkbox("🔊 语音反馈", value=True)
        
        st.divider()
        
        # 工作流状态
        st.subheader("📊 当前状态")
        stage_labels = {
            "idle": "⏸️ 待命",
            "analyzing": "🔍 分析中...",
            "guiding": "🎯 引导中",
            "countdown": "⏱️ 倒计时",
            "captured": "✅ 已拍摄"
        }
        st.write(stage_labels.get(st.session_state.workflow_stage, "未知"))
        
        # 拍照历史
        st.divider()
        st.subheader("📷 拍照历史")
        if st.session_state.captured_photos:
            for photo, ts in st.session_state.captured_photos[-3:]:
                st.image(photo, caption=ts, use_container_width=True)
        else:
            st.caption("还没有拍摄照片")
    
    # 主区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 实时画面")
        
        # WebRTC
        ctx = webrtc_streamer(
            key="pose-detection-v2",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=PoseVideoProcessor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # 操作按钮
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🚀 开始引导", use_container_width=True, type="primary",
                        disabled=st.session_state.workflow_stage != "idle"):
                start_guidance_workflow()
        
        with col_btn2:
            if st.button("📸 立即拍照", use_container_width=True,
                        disabled=st.session_state.workflow_stage not in ["guiding", "idle"]):
                trigger_capture()
        
        with col_btn3:
            if st.button("� 重新开始", use_container_width=True):
                reset_workflow()
    
    with col2:
        st.subheader("📊 匹配状态")
        
        # 根据工作流阶段显示不同内容
        stage = st.session_state.workflow_stage
        
        if stage == "idle":
            st.markdown("""
            <div class="guide-step">
                <h4>👋 准备开始</h4>
                <p>点击「开始引导」让 AI 分析场景并推荐最佳姿势</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif stage == "analyzing":
            st.markdown("""
            <div class="analyzing-box">
                <h3>🔍 正在分析环境光线...</h3>
                <p>AI 正在识别场景特征</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 同时显示默认姿势提示
            st.info("💡 先保持自然站立姿势，稍后会显示最佳建议")
            
        elif stage == "guiding":
            # 分数显示
            score = st.session_state.match_score
            score_class = "score-high" if score >= 70 else "score-medium" if score >= 50 else "score-low"
            
            st.markdown(f'<div class="score-display {score_class}">{score:.0f}%</div>', 
                       unsafe_allow_html=True)
            
            # 进度条
            st.progress(min(score / 100, 1.0))
            
            # 反馈
            st.markdown(f"""
            <div class="feedback-box">
                <strong>💬 AI 指导：</strong><br>
                {st.session_state.feedback}
            </div>
            """, unsafe_allow_html=True)
            
            # AI 分析结果
            if st.session_state.ai_analysis:
                with st.expander("🤖 AI 场景分析", expanded=False):
                    analysis = st.session_state.ai_analysis
                    st.write(f"**场景：** {analysis.get('scene_analysis', 'N/A')}")
                    st.write(f"**建议：** {analysis.get('composition_advice', 'N/A')}")
            
            # 自动拍照检测
            check_auto_capture()
            
        elif stage == "countdown":
            countdown = st.session_state.countdown_value
            st.markdown(f'<div class="countdown">{countdown}</div>', unsafe_allow_html=True)
            
        elif stage == "captured":
            st.markdown('<div class="perfect-banner">✨ PERFECT! ✨</div>', unsafe_allow_html=True)
            st.balloons()
            
            if st.session_state.captured_photos:
                latest = st.session_state.captured_photos[-1]
                st.image(latest[0], caption=f"📸 {latest[1]}", use_container_width=True)
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 🎬 拍照流程
        1. **开启摄像头** - 点击 START 允许摄像头权限
        2. **开始引导** - AI 分析场景并推荐姿势
        3. **跟随引导** - 移动到绿色虚线框位置
        4. **自动拍照** - 匹配度 ≥85% 持续 2 秒自动拍照
        
        ### 🎨 图例
        - 🟢 **绿色虚线** - 目标姿势引导框
        - ⚪ **白色实线** - 你的实时骨骼
        - 📊 **进度条** - 姿势匹配程度
        """)


def start_guidance_workflow():
    """开始引导工作流"""
    st.session_state.workflow_stage = "analyzing"
    st.session_state.is_analyzing = True
    
    # 语音提示
    if st.session_state.voice_enabled:
        speak_text_js("正在分析场景，请稍候")
    
    # 模拟 API 调用延迟后切换到引导模式
    # 实际项目中这里应该异步调用 GPT-4o
    time.sleep(0.5)  # 模拟
    
    # 模拟 AI 分析结果
    st.session_state.ai_analysis = {
        "scene_analysis": "室内光线良好，背景简洁",
        "recommended_pose_id": st.session_state.target_pose,
        "composition_advice": "保持自然站姿，面向镜头微笑",
        "voice_feedback": "很好，光线不错，请保持自然站姿"
    }
    
    st.session_state.workflow_stage = "guiding"
    st.session_state.is_analyzing = False
    
    if st.session_state.voice_enabled:
        speak_text_js(st.session_state.ai_analysis.get("voice_feedback", ""))
    
    st.rerun()


def check_auto_capture():
    """检查是否触发自动拍照"""
    score = st.session_state.match_score
    threshold = st.session_state.auto_capture_threshold
    
    if score >= threshold:
        if st.session_state.high_score_start is None:
            st.session_state.high_score_start = time.time()
        elif time.time() - st.session_state.high_score_start >= 2.0:
            trigger_countdown()
    else:
        st.session_state.high_score_start = None


def trigger_countdown():
    """触发倒计时"""
    st.session_state.workflow_stage = "countdown"
    
    for i in [3, 2, 1]:
        st.session_state.countdown_value = i
        if st.session_state.voice_enabled:
            speak_text_js(str(i))
        time.sleep(1)
    
    trigger_capture()


def trigger_capture():
    """执行拍照"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # 这里应该从视频流获取帧
    # 简化版本：记录时间戳
    st.session_state.captured_photos.append((None, timestamp))
    st.session_state.workflow_stage = "captured"
    st.session_state.high_score_start = None
    
    if st.session_state.voice_enabled:
        speak_text_js("拍摄成功！太棒了！")
    
    st.rerun()


def reset_workflow():
    """重置工作流"""
    st.session_state.workflow_stage = "idle"
    st.session_state.match_score = 0.0
    st.session_state.feedback = "准备好后点击「开始引导」"
    st.session_state.ai_analysis = None
    st.session_state.high_score_start = None
    st.session_state.countdown_value = None
    st.session_state.show_perfect = False
    st.rerun()


if __name__ == "__main__":
    main()
