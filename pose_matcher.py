"""
AI Photography Director - 模块二：The Matcher (姿态匹配器)
使用 MediaPipe Pose 关键点进行实时姿态比较
基于余弦相似度算法，无需调用 LLM，速度极快
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ==================== 常量定义 ====================

# MediaPipe Pose 上半身关键点索引
class PoseLandmark(Enum):
    """MediaPipe Pose 关键点索引 (只保留我们需要的)"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24


# 我们关注的上半身关键点
UPPER_BODY_LANDMARKS = [
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_WRIST,
]


@dataclass
class Landmark:
    """单个关键点"""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class PoseVector:
    """归一化后的姿态向量"""
    left_upper_arm: np.ndarray   # 肩膀 -> 手肘
    right_upper_arm: np.ndarray
    left_forearm: np.ndarray     # 手肘 -> 手腕
    right_forearm: np.ndarray


# ==================== 预定义的目标姿势 ====================

# 这些是硬编码的"标准姿势"向量，用于快速比较
# 向量格式：[dx, dy] 相对于归一化原点
TARGET_POSES: Dict[str, Dict] = {
    "standing_casual": {
        "description": "自然站立，双手自然下垂",
        "left_upper_arm": np.array([0.0, 0.5]),      # 手臂自然下垂
        "right_upper_arm": np.array([0.0, 0.5]),
        "left_forearm": np.array([0.0, 0.4]),
        "right_forearm": np.array([0.0, 0.4]),
    },
    "leaning_wall": {
        "description": "倚靠墙壁，一手叉腰",
        "left_upper_arm": np.array([-0.3, 0.3]),     # 左手叉腰
        "right_upper_arm": np.array([0.0, 0.5]),     # 右手自然下垂
        "left_forearm": np.array([-0.2, -0.2]),      # 手肘弯曲向内
        "right_forearm": np.array([0.0, 0.4]),
    },
    "sitting_coffee": {
        "description": "坐着喝东西，双手在胸前",
        "left_upper_arm": np.array([-0.2, 0.3]),
        "right_upper_arm": np.array([0.2, 0.3]),
        "left_forearm": np.array([0.3, -0.1]),       # 手臂弯曲向中间
        "right_forearm": np.array([-0.3, -0.1]),
    },
    "walking_away": {
        "description": "背影行走，自然摆臂",
        "left_upper_arm": np.array([0.1, 0.5]),      # 一前一后
        "right_upper_arm": np.array([-0.1, 0.5]),
        "left_forearm": np.array([0.1, 0.3]),
        "right_forearm": np.array([-0.1, 0.3]),
    },
}


# ==================== 核心算法函数 ====================

def normalize_landmarks(landmarks: List[Landmark]) -> List[Landmark]:
    """
    归一化关键点坐标
    以双肩中点为原点，肩宽为单位长度
    
    Args:
        landmarks: MediaPipe 返回的 33 个关键点列表
    
    Returns:
        归一化后的关键点列表
    """
    if len(landmarks) < 17:  # 至少需要到手腕的点
        raise ValueError("Not enough landmarks provided")
    
    # 获取双肩
    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
    
    # 计算原点（双肩中点）
    origin_x = (left_shoulder.x + right_shoulder.x) / 2
    origin_y = (left_shoulder.y + right_shoulder.y) / 2
    origin_z = (left_shoulder.z + right_shoulder.z) / 2
    
    # 计算肩宽作为归一化因子
    shoulder_width = np.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2
    )
    
    # 防止除以零
    if shoulder_width < 0.01:
        shoulder_width = 0.01
    
    # 归一化所有点
    normalized = []
    for lm in landmarks:
        normalized.append(Landmark(
            x=(lm.x - origin_x) / shoulder_width,
            y=(lm.y - origin_y) / shoulder_width,
            z=(lm.z - origin_z) / shoulder_width,
            visibility=lm.visibility
        ))
    
    return normalized


def extract_pose_vectors(landmarks: List[Landmark]) -> PoseVector:
    """
    从关键点提取姿态向量（肢体方向）
    
    Args:
        landmarks: 归一化后的关键点
    
    Returns:
        包含四个肢体向量的 PoseVector
    """
    # 获取关键点
    left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[PoseLandmark.RIGHT_WRIST.value]
    
    # 计算向量（从关节到关节）
    left_upper_arm = np.array([
        left_elbow.x - left_shoulder.x,
        left_elbow.y - left_shoulder.y
    ])
    right_upper_arm = np.array([
        right_elbow.x - right_shoulder.x,
        right_elbow.y - right_shoulder.y
    ])
    left_forearm = np.array([
        left_wrist.x - left_elbow.x,
        left_wrist.y - left_elbow.y
    ])
    right_forearm = np.array([
        right_wrist.x - right_elbow.x,
        right_wrist.y - right_elbow.y
    ])
    
    return PoseVector(
        left_upper_arm=left_upper_arm,
        right_upper_arm=right_upper_arm,
        left_forearm=left_forearm,
        right_forearm=right_forearm
    )


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    
    Args:
        v1, v2: 输入向量
    
    Returns:
        余弦相似度 (-1 到 1)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def calculate_pose_similarity(
    user_landmarks: List[Landmark],
    target_pose_id: str
) -> float:
    """
    计算用户姿态与目标姿态的相似度
    
    Args:
        user_landmarks: 用户的 MediaPipe Pose 关键点（33个）
        target_pose_id: 目标姿势ID（如 "standing_casual"）
    
    Returns:
        相似度分数 (0-100)
    """
    if target_pose_id not in TARGET_POSES:
        raise ValueError(f"Unknown pose ID: {target_pose_id}")
    
    target = TARGET_POSES[target_pose_id]
    
    # 归一化用户关键点
    normalized = normalize_landmarks(user_landmarks)
    
    # 提取用户姿态向量
    user_vectors = extract_pose_vectors(normalized)
    
    # 计算每个肢体的余弦相似度
    similarities = [
        cosine_similarity(user_vectors.left_upper_arm, target["left_upper_arm"]),
        cosine_similarity(user_vectors.right_upper_arm, target["right_upper_arm"]),
        cosine_similarity(user_vectors.left_forearm, target["left_forearm"]),
        cosine_similarity(user_vectors.right_forearm, target["right_forearm"]),
    ]
    
    # 加权平均（上臂权重略高）
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
    
    # 转换到 0-100 分数
    # 余弦相似度范围 [-1, 1]，映射到 [0, 100]
    score = (weighted_similarity + 1) / 2 * 100
    
    return round(score, 1)


def calculate_pose_similarity_with_target(
    user_landmarks: List[Landmark],
    target_landmarks: List[Landmark]
) -> float:
    """
    计算用户姿态与另一个姿态的相似度（用于自定义目标）
    
    Args:
        user_landmarks: 用户的关键点
        target_landmarks: 目标姿态的关键点
    
    Returns:
        相似度分数 (0-100)
    """
    # 归一化两组关键点
    user_normalized = normalize_landmarks(user_landmarks)
    target_normalized = normalize_landmarks(target_landmarks)
    
    # 提取向量
    user_vectors = extract_pose_vectors(user_normalized)
    target_vectors = extract_pose_vectors(target_normalized)
    
    # 计算相似度
    similarities = [
        cosine_similarity(user_vectors.left_upper_arm, target_vectors.left_upper_arm),
        cosine_similarity(user_vectors.right_upper_arm, target_vectors.right_upper_arm),
        cosine_similarity(user_vectors.left_forearm, target_vectors.left_forearm),
        cosine_similarity(user_vectors.right_forearm, target_vectors.right_forearm),
    ]
    
    weights = [0.3, 0.3, 0.2, 0.2]
    weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
    score = (weighted_similarity + 1) / 2 * 100
    
    return round(score, 1)


# ==================== 反馈生成函数 ====================

def get_feedback_instruction(
    score: float,
    user_landmarks: List[Landmark],
    target_pose_id: str
) -> Optional[str]:
    """
    根据姿态分数生成反馈指令
    
    Args:
        score: 姿态相似度分数 (0-100)
        user_landmarks: 用户关键点
        target_pose_id: 目标姿势ID
    
    Returns:
        反馈指令字符串，如果分数足够高则返回 None
    """
    # 分数够高，不需要调整
    if score >= 80:
        return None
    
    if score >= 60:
        return "差不多了！再微调一下就完美了～"
    
    # 分数低，需要详细反馈
    if target_pose_id not in TARGET_POSES:
        return "请调整一下姿势"
    
    target = TARGET_POSES[target_pose_id]
    normalized = normalize_landmarks(user_landmarks)
    user_vectors = extract_pose_vectors(normalized)
    
    feedbacks = []
    
    # 检查左臂
    left_arm_sim = cosine_similarity(user_vectors.left_upper_arm, target["left_upper_arm"])
    if left_arm_sim < 0.5:
        # 判断左臂位置
        if user_vectors.left_upper_arm[1] > target["left_upper_arm"][1] + 0.2:
            feedbacks.append("左手抬高一点")
        elif user_vectors.left_upper_arm[1] < target["left_upper_arm"][1] - 0.2:
            feedbacks.append("左手放低一点")
        else:
            feedbacks.append("左手位置调整一下")
    
    # 检查右臂
    right_arm_sim = cosine_similarity(user_vectors.right_upper_arm, target["right_upper_arm"])
    if right_arm_sim < 0.5:
        if user_vectors.right_upper_arm[1] > target["right_upper_arm"][1] + 0.2:
            feedbacks.append("右手抬高一点")
        elif user_vectors.right_upper_arm[1] < target["right_upper_arm"][1] - 0.2:
            feedbacks.append("右手放低一点")
        else:
            feedbacks.append("右手位置调整一下")
    
    # 检查前臂
    left_forearm_sim = cosine_similarity(user_vectors.left_forearm, target["left_forearm"])
    if left_forearm_sim < 0.3:
        feedbacks.append("左手肘弯曲角度调整一下")
    
    right_forearm_sim = cosine_similarity(user_vectors.right_forearm, target["right_forearm"])
    if right_forearm_sim < 0.3:
        feedbacks.append("右手肘弯曲角度调整一下")
    
    if not feedbacks:
        return "稍微调整一下姿势，快到位了！"
    
    return "、".join(feedbacks[:2])  # 最多返回2个提示，避免信息过载


def get_detailed_analysis(
    user_landmarks: List[Landmark],
    target_pose_id: str
) -> Dict:
    """
    获取详细的姿态分析报告
    
    Args:
        user_landmarks: 用户关键点
        target_pose_id: 目标姿势ID
    
    Returns:
        详细分析字典
    """
    if target_pose_id not in TARGET_POSES:
        return {"error": f"Unknown pose: {target_pose_id}"}
    
    target = TARGET_POSES[target_pose_id]
    normalized = normalize_landmarks(user_landmarks)
    user_vectors = extract_pose_vectors(normalized)
    
    # 各部位分数
    left_upper_score = (cosine_similarity(user_vectors.left_upper_arm, target["left_upper_arm"]) + 1) / 2 * 100
    right_upper_score = (cosine_similarity(user_vectors.right_upper_arm, target["right_upper_arm"]) + 1) / 2 * 100
    left_forearm_score = (cosine_similarity(user_vectors.left_forearm, target["left_forearm"]) + 1) / 2 * 100
    right_forearm_score = (cosine_similarity(user_vectors.right_forearm, target["right_forearm"]) + 1) / 2 * 100
    
    overall_score = calculate_pose_similarity(user_landmarks, target_pose_id)
    
    return {
        "overall_score": overall_score,
        "target_pose": target_pose_id,
        "target_description": target["description"],
        "breakdown": {
            "left_upper_arm": round(left_upper_score, 1),
            "right_upper_arm": round(right_upper_score, 1),
            "left_forearm": round(left_forearm_score, 1),
            "right_forearm": round(right_forearm_score, 1),
        },
        "feedback": get_feedback_instruction(overall_score, user_landmarks, target_pose_id),
        "is_match": overall_score >= 70
    }


# ==================== MediaPipe 集成辅助函数 ====================

def landmarks_from_mediapipe(mp_landmarks) -> List[Landmark]:
    """
    将 MediaPipe 的 landmark 对象转换为我们的 Landmark 列表
    
    Args:
        mp_landmarks: mediapipe.python.solutions.pose.PoseLandmark 结果
    
    Returns:
        Landmark 对象列表
    """
    return [
        Landmark(
            x=lm.x,
            y=lm.y,
            z=lm.z,
            visibility=lm.visibility
        )
        for lm in mp_landmarks.landmark
    ]


def landmarks_from_dict(landmarks_dict: List[Dict]) -> List[Landmark]:
    """
    将字典列表转换为 Landmark 列表（用于 API 传输）
    
    Args:
        landmarks_dict: [{"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9}, ...]
    
    Returns:
        Landmark 对象列表
    """
    return [
        Landmark(
            x=lm.get("x", 0),
            y=lm.get("y", 0),
            z=lm.get("z", 0),
            visibility=lm.get("visibility", 1.0)
        )
        for lm in landmarks_dict
    ]


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 创建模拟的用户关键点（33个，这里只填充我们需要的）
    mock_landmarks = [Landmark(x=0, y=0, z=0) for _ in range(33)]
    
    # 模拟一个"自然站立"的姿势
    mock_landmarks[11] = Landmark(x=0.4, y=0.3, z=0)   # 左肩
    mock_landmarks[12] = Landmark(x=0.6, y=0.3, z=0)   # 右肩
    mock_landmarks[13] = Landmark(x=0.35, y=0.5, z=0)  # 左肘
    mock_landmarks[14] = Landmark(x=0.65, y=0.5, z=0)  # 右肘
    mock_landmarks[15] = Landmark(x=0.35, y=0.7, z=0)  # 左腕
    mock_landmarks[16] = Landmark(x=0.65, y=0.7, z=0)  # 右腕
    
    # 测试各个姿势的匹配度
    print("=" * 50)
    print("姿态匹配器测试")
    print("=" * 50)
    
    for pose_id in TARGET_POSES.keys():
        score = calculate_pose_similarity(mock_landmarks, pose_id)
        feedback = get_feedback_instruction(score, mock_landmarks, pose_id)
        print(f"\n姿势: {pose_id}")
        print(f"  分数: {score}/100")
        print(f"  反馈: {feedback or '完美！'}")
    
    # 详细分析
    print("\n" + "=" * 50)
    print("详细分析 (standing_casual)")
    print("=" * 50)
    analysis = get_detailed_analysis(mock_landmarks, "standing_casual")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
