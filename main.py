"""
AI Photography Director - FastAPI Backend
黑客松项目：AI拍照助手 - 模块一：Director Agent
支持多种国内外大模型 API
"""

import os
import json
import base64
import httpx
import shutil
from typing import Optional, Literal, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = FastAPI(
    title="AI Photography Director",
    description="AI拍照助手后端服务 - 帮助不会拍照的人群拍出美美的照片",
    version="1.0.0"
)

# CORS 配置 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic 模型定义 ====================

class ImageAnalysisRequest(BaseModel):
    """前端请求模型"""
    image_base64: str = Field(..., description="Base64编码的图片数据 (支持 JPEG/WebP/PNG)")
    image_format: Optional[str] = Field("jpeg", description="图片格式: jpeg, webp, png")
    user_message: Optional[str] = Field(None, description="用户的语音/文字输入（可选）")
    provider: Optional[str] = Field(None, description="指定使用的模型提供商（可选）")


class StyleLearningRequest(BaseModel):
    """风格学习请求"""
    images: List[str] = Field(..., description="用户上传的参考图Base64列表")


class DirectorResponse(BaseModel):
    """AI导演返回的分析结果"""
    detected_scene: Optional[str] = Field(None, description="检测到的场景类型")
    scene_analysis: Optional[str] = Field(None, description="场景分析")
    recommended_pose_id: Optional[str] = Field(None, description="推荐的姿势ID")
    voice_feedback: Optional[str] = Field(None, description="语音反馈")
    direction_guidance: Optional[str] = Field("none", description="构图引导")
    framing_type: Optional[str] = Field("full_body", description="取景框类型: selfie/upper_body/full_body")
    action: str = Field("continue", description="用户意图：capture/change_pose/continue/talk")


class APIResponse(BaseModel):
    """统一API响应格式"""
    success: bool
    data: Optional[DirectorResponse | dict] = None  # 支持返回字典(如风格学习结果)
    error: Optional[str] = None
    provider: Optional[str] = None  # 实际使用的模型提供商
    style_profile: Optional[str] = None # 风格描述


# ==================== 全局变量 ====================

# 存储用户的风格画像 (暂存内存，生产环境应存数据库)
USER_STYLE_PROFILE = ""

# 本地存储路径配置
STYLE_DIR = os.path.join(os.path.dirname(__file__), "user_styles")
STYLE_IMAGES_DIR = os.path.join(STYLE_DIR, "images")
STYLE_PROFILE_PATH = os.path.join(STYLE_DIR, "profile.json")


# ==================== 本地存储辅助函数 ====================

def ensure_style_dir():
    """确保风格存储目录存在"""
    os.makedirs(STYLE_IMAGES_DIR, exist_ok=True)
    print(f"✅ 风格存储目录已就绪: {STYLE_DIR}")


def load_user_profile():
    """从本地加载用户风格画像"""
    global USER_STYLE_PROFILE
    if os.path.exists(STYLE_PROFILE_PATH):
        try:
            with open(STYLE_PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                USER_STYLE_PROFILE = data.get("style_description", "")
                print(f"✅ 加载用户风格画像: {USER_STYLE_PROFILE[:100]}...")
        except Exception as e:
            print(f"⚠️ 加载风格画像失败: {e}")
    else:
        print("ℹ️ 未找到风格画像文件，将在首次学习后创建")


def save_user_profile():
    """保存用户风格画像到本地"""
    try:
        data = {
            "style_description": USER_STYLE_PROFILE,
            "updated_at": str(os.path.getmtime(STYLE_PROFILE_PATH)) if os.path.exists(STYLE_PROFILE_PATH) else "new"
        }
        with open(STYLE_PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ 风格画像已保存: {STYLE_PROFILE_PATH}")
    except Exception as e:
        print(f"❌ 保存风格画像失败: {e}")


def save_style_image(base64_data: str, index: int = 0) -> str:
    """保存单张风格参考图到本地"""
    try:
        import uuid
        from datetime import datetime
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"style_{timestamp}_{index}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(STYLE_IMAGES_DIR, filename)
        
        # 解码并保存
        image_data = base64.b64decode(base64_data)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        print(f"✅ 风格图片已保存: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ 保存风格图片失败: {e}")
        return ""


# ==================== System Prompt ====================

DIRECTOR_SYSTEM_PROMPT = """# 身份设定

    你是一位专业的摄影美学顾问，拥有敏锐的艺术直觉和高情商的沟通技巧。
    你的目标是帮助用户拍出最完美的照片，不仅提供构图建议，更要提供情绪价值。

# 视觉感知与构图意识 (Composition Awareness)

你必须识别当前画面中的人物景别：
- **selfie**: 特写或自拍（画面大部分是头部和肩膀）。
- **upper_body**: 中景或半身（画面包含腰部以上）。
- **full_body**: 全景或全身（画面包含整个人或大部分身体）。

# 动作判断与姿势推荐 (Pose Recommendation)

- **CRITICAL**: 推荐的姿势ID（recommended_pose_id）必须与当前景别匹配！
- **自拍模式 (selfie_*)**: 建议头部的倾斜角度、肩膀的姿态、手在脸部附近的位置。
- **半身模式 (upper_*)**: 建议双手的摆放地点、转动身体的角度。
- **全身模式 (full_*)**: 建议双腿的跨度、整体身体的重心偏移。

# 构图与位移引导 (Directional Guidance)

你必须通过分析人物在画面中的位置，给出具体的手机调整建议：
- **direction_guidance** 字段必须返回以下值之一：
  - `move_left` / `move_right`: 手机平移。
  - `move_up` / `move_down`: 手机上下平移（注意：脚部被切断时建议 `move_down` 给脚留空）。
  - `tilt_up` / `tilt_down`: 手机俯仰（拍长腿建议 `tilt_up` 仰拍）。
  - `zoom_in` / `zoom_out`: 靠近或远离。
  - `none`: 构图完美，无需调整。

    尊重用户的即时需求，但要用自然的口语确认：
    - 用户想拍照时，自然地回应并执行 `action: "capture"`。
    - 用户想换姿势时，提供新的建议并执行 `action: "change_pose"`。

# 用户个人风格 (Personal Style)
{user_style_context}

# 沟通风格 (Natural & Fluid)
- **拒绝机器味**：绝对不要说“好的，正在...”、“收到指令”这种机器语言。
- **生动自然**：像闺蜜或好哥们一样聊天，使用感叹词（哇、太棒了、稍微往左一点点）。
- **多变性**：每次的反馈都要有所不同，不要重复一样的话术。
- **简短有力**：在指导动作时要短促清晰（例如：“头歪一点”、“看镜头微笑”），在夸奖时要真诚热情。

# 输出格式

必须返回严格的JSON：
{
  "detected_scene": "场景类型 (如: 室内/街头/自拍)",
  "framing_type": "selfie/upper_body/full_body",
  "direction_guidance": "move_left/move_right/move_up/move_down/tilt_up/tilt_down/zoom_in/zoom_out/none",
  "scene_analysis": "对画面的简短描述，包括建议的距离",
  "recommended_pose_id": "具体姿势ID",
  "voice_feedback": "一句简短、同步用户意图的引导语",
  "action": "capture/change_pose/talk/continue"
}"""


# ==================== 模型提供商配置 ====================

MODEL_PROVIDERS = {
    "minimax": {
        "name": "MiniMax (abab6.5)",
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
    },
    "openai": {
        "name": "OpenAI GPT-4o",
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
    },
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "qwen": {
        "name": "阿里通义千问 (Qwen-VL)",
        "env_key": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "zhipu": {
        "name": "智谱 GLM-4V",
        "env_key": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "moonshot": {
        "name": "月之暗面 Kimi",
        "env_key": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
    },
    "baichuan": {
        "name": "百川大模型",
        "env_key": "BAICHUAN_API_KEY",
        "base_url": "https://api.baichuan-ai.com/v1",
    },
    "deepseek": {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "stepfun": {
        "name": "阶跃星辰 Step",
        "env_key": "STEPFUN_API_KEY",
        "base_url": "https://api.stepfun.com/v1",
    },
    "minimax": {
        "name": "MiniMax (abab6.5)",
        "env_key": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.chat/v1",
        "mcp_enabled": True,  # 支持 MCP 协议
    },
}


# ==================== MCP 协议配置 ====================

class MCPTool:
    """MCP 工具定义"""
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }


class MCPServerConfig:
    """MCP 服务器配置"""
    def __init__(self, name: str, url: str, tools: list):
        self.name = name
        self.url = url
        self.tools = tools

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "url": self.url,
            "tools": [t.to_dict() for t in self.tools]
        }


# 图片理解 MCP 工具
UNDERSTAND_IMAGE_TOOL = MCPTool(
    name="understand_image",
    description="分析图片内容，识别场景、光线、人物姿态，并给出拍照建议",
    input_schema={
        "type": "object",
        "properties": {
            "image_base64": {
                "type": "string",
                "description": "Base64 编码的图片数据"
            },
            "user_message": {
                "type": "string",
                "description": "用户的语音/文字输入，可选"
            }
        },
        "required": ["image_base64"]
    }
)

# MCP 服务器配置（可以连接到外部 MCP 服务器）
MCP_SERVER_CONFIG = MCPServerConfig(
    name="image_understanding",
    url=os.getenv("MCP_SERVER_URL", "http://localhost:3000/mcp"),
    tools=[UNDERSTAND_IMAGE_TOOL]
)


async def call_mcp_understand_image(image_base64: str, user_message: Optional[str] = None) -> dict:
    """
    通过 MCP 协议调用图片理解工具
    
    支持两种模式：
    1. 直接调用：图片数据直接发送给 MCP 服务器
    2. 远程 MCP：连接到外部 MCP 服务器
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    if mcp_server_url:
        # 模式1：连接到外部 MCP 服务器
        print(f"🔗 通过 MCP 服务器调用: {mcp_server_url}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{mcp_server_url}/tools/understand_image",
                json={
                    "image_base64": image_base64,
                    "user_message": user_message
                }
            )
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"MCP error: {response.text}")
            return response.json()
    else:
        # 模式2：本地 MCP 处理（这里可以集成其他图片理解服务）
        # 如果没有配置 MCP 服务器，使用默认的 MiniMax API
        print("⚠️ 未配置 MCP 服务器，使用 MiniMax API")
        return await call_minimax_vision(image_base64, user_message)


# ==================== Vision API 调用 ====================

async def call_minimax_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg", style_context: str = "") -> dict:
    """调用 MiniMax abab6.5 Vision API"""
    
    api_key = os.getenv("MINIMAX_API_KEY")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY not configured")
    
    # 注入风格上下文到 System Prompt
    # FIX: 使用 replace 而不是 format，避免 JSON 中的花括号被误解析
    system_content = DIRECTOR_SYSTEM_PROMPT.replace("{user_style_context}", style_context)

    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "abab6.5s-chat",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    if group_id:
        url = f"https://api.minimax.chat/v1/text/chatcompletion_v2?GroupId={group_id}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"MiniMax API error: {response.text}")
        return response.json()


async def call_minimax_style_learning(images: List[str]) -> str:
    """调用 MiniMax 批量分析图片风格"""
    api_key = os.getenv("MINIMAX_API_KEY")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")
    
    # 构造 Prompt
    prompt = """请作为一位资深摄影师，分析这些照片的共同视觉风格。
    请关注：
    1. 构图习惯（如居中、三分、留白、特写）
    2. 光影偏好（如逆光、高对比、柔光、硬光）
    3. 色调氛围（如冷色调、暖色调、黑白、高饱和、胶片感）
    
    请用一段简练的话总结这位用户的“摄影审美偏好”。例如：“用户偏爱高对比度的黑白街头摄影，喜欢捕捉光影的几何形状。”
    不要分点，直接输出一段描述。"""

    # 构造 content
    content = [{"type": "text", "text": prompt}]
    for img_b64 in images:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    payload = {
        "model": "abab6.5s-chat",
        "messages": [
            {"role": "user", "content": content}
        ],
        "max_tokens": 300,
        "temperature": 0.6
    }
    
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    if group_id:
        url = f"https://api.minimax.chat/v1/text/chatcompletion_v2?GroupId={group_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # 提取回复
        try:
             return result['choices'][0]['message']['content']
        except:
             return "无法分析风格"


async def call_openai_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用 OpenAI GPT-4o Vision API"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    mime_type = f"image/{image_format or 'jpeg'}"
    user_content = [
        {"type": "text", "text": f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}", "detail": "low"}}
    ]
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {response.text}")
        return response.json()


async def call_gemini_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用 Google Gemini Vision API"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": DIRECTOR_SYSTEM_PROMPT + "\n\n" + user_text},
                {"inline_data": {"mime_type": mime_type, "data": image_base64}}
            ]
        }],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Gemini API error: {response.text}")
        return response.json()


async def call_qwen_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用阿里通义千问 Qwen-VL-Plus/Max Vision API"""
    
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="QWEN_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "qwen-vl-plus",  # 或 qwen-vl-max
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Qwen API error: {response.text}")
        return response.json()


async def call_zhipu_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用智谱 GLM-4V Vision API（免费的 flash 版本）"""
    
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ZHIPU_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "glm-4v-flash",  # 免费版本！
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Zhipu API error: {response.text}")
        return response.json()


async def call_moonshot_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用月之暗面 Kimi Vision API (moonshot-v1-8k-vision-preview)"""
    
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MOONSHOT_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "moonshot-v1-8k-vision-preview",
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Moonshot API error: {response.text}")
        return response.json()


async def call_stepfun_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用阶跃星辰 Step Vision API (step-1v-8k)"""
    
    api_key = os.getenv("STEPFUN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="STEPFUN_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "step-1v-8k",
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.stepfun.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"StepFun API error: {response.text}")
        return response.json()


async def call_baichuan_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用百川大模型 Vision API"""
    
    api_key = os.getenv("BAICHUAN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="BAICHUAN_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "Baichuan4-Turbo",
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.baichuan-ai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Baichuan API error: {response.text}")
        return response.json()


async def call_deepseek_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg") -> dict:
    """调用 DeepSeek Vision API (deepseek-vl)"""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")
    
    user_text = f"用户说：{user_message}" if user_message else "请分析这张图片并给出拍照建议。"
    mime_type = f"image/{image_format or 'jpeg'}"
    
    payload = {
        "model": "deepseek-vl",
        "messages": [
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"DeepSeek API error: {response.text}")
        return response.json()


# ==================== 交替调用双模型 ====================

import asyncio
from datetime import datetime

# 存储上次使用的模型
_last_model_used = {"minimax": False}  # False = 上次用 minimax, True = 上次用 stepfun


async def call_race_vision(image_base64: str, user_message: Optional[str] = None, image_format: str = "jpeg", style_context: str = "") -> dict:
    """
    【极速模式】同时请求 MiniMax 和 StepFun，谁先返回用谁
    """
    print(f"🏎️ 启动双模型竞速: MiniMax vs StepFun")
    
    minimax_key = os.getenv("MINIMAX_API_KEY")
    stepfun_key = os.getenv("STEPFUN_API_KEY")

    # 1. 定义请求任务
    async def fast_minimax():
        return await call_minimax_vision(image_base64, user_message, image_format, style_context)

    async def fast_stepfun():
        return await call_stepfun_vision(image_base64, user_message, image_format)

    # 2. 创建任务
    tasks = []
    if minimax_key: tasks.append(asyncio.create_task(fast_minimax()))
    if stepfun_key: tasks.append(asyncio.create_task(fast_stepfun()))
    
    if not tasks:
        raise HTTPException(status_code=500, detail="No API keys for race mode")

    # 3. 竞速等待 (改进版：等待首个成功结果)
    try:
        # 使用 asyncio.as_completed 迭代完成的任务
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                try:
                    result = task.result()
                    # 成功获取结果！
                    print("🏆 竞速胜出: " + ("MiniMax" if task.get_name() == "minimax" else "StepFun"))
                    
                    # 取消剩余任务
                    for p in pending:
                        p.cancel()
                    return result
                    
                except Exception as e:
                    import traceback
                    print(f"⚠️ 竞速中一员失败: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    # 继续等待其他任务
                    continue
        
        # 如果所有任务都失败了
        raise HTTPException(status_code=500, detail="All race models failed.")

    except Exception as e:
        print(f"❌ 竞速模式全军覆没: {e}")
        traceback.print_exc()
        # 最后的兜底
        if minimax_key: return await call_minimax_vision(image_base64, user_message, image_format, style_context)
        raise


# ==================== 响应解析 ====================

def parse_llm_response(response: dict, provider: str = "openai") -> DirectorResponse:
    """解析 LLM 返回的 JSON"""
    
    import re
    
    try:
        if provider == "gemini":
            content = response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            content = response["choices"][0]["message"]["content"]
        
        print(f"📝 AI 原始返回: {content[:200]}...")
        
        # 尝试匹配 JSON 对象
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        # 清理可能的 markdown 标记
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # 如果清理后不是以 { 开头，尝试构造一个完整的 JSON
        if not content.startswith("{"):
            print(f"⚠️ AI 返回格式异常，尝试解析...")
            # 尝试找到 key: value 对
            detected_scene_match = re.search(r'"detected_scene"\s*:\s*"([^"]*)"', content)
            scene_analysis_match = re.search(r'"scene_analysis"\s*:\s*"([^"]*)"', content)
            framing_type_match = re.search(r'"framing_type"\s*:\s*"([^"]*)"', content)
            voice_feedback_match = re.search(r'"voice_feedback"\s*:\s*"([^"]*)"', content)
            recommended_pose_match = re.search(r'"recommended_pose_id"\s*:\s*"([^"]*)"', content)
            action_match = re.search(r'"action"\s*:\s*"([^"]*)"', content)
            
            if any([detected_scene_match, scene_analysis_match, voice_feedback_match]):
                # 找到了部分字段，构造完整 JSON
                data = {
                    "detected_scene": detected_scene_match.group(1) if detected_scene_match else None,
                    "scene_analysis": scene_analysis_match.group(1) if scene_analysis_match else None,
                    "voice_feedback": voice_feedback_match.group(1) if voice_feedback_match else None,
                    "recommended_pose_id": recommended_pose_match.group(1) if recommended_pose_match else None,
                    "framing_type": framing_type_match.group(1) if framing_type_match else None,
                    "action": action_match.group(1) if action_match else "continue"
                }
                print(f"🔧 手动构造响应: {data}")
                return DirectorResponse(**data)
            else:
                # 完全没有找到关键字段，返回默认响应
                print(f"⚠️ 无法解析 AI 返回，使用默认响应")
                return DirectorResponse(
                    action="continue"
                )
        
        # 解析 JSON
        data = json.loads(content)
        
        # 验证并返回
        return DirectorResponse(**data)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw Content: {content[:200]}")
        # 返回默认响应而不是抛出错误
        # 返回默认响应而不是抛出错误
        return DirectorResponse(
            action="continue"
        )
    except ValidationError as e:
        print(f"❌ Pydantic Validation Error: {e}")
        print(f"Raw Content: {content[:200]}")
        # 返回默认响应
        # 返回默认响应
        return DirectorResponse(
            action="continue"
        )
    except KeyError as e:
        print(f"❌ KeyError in Response: {e}")
        print(f"Full Response: {response}")
        # 返回默认响应
        # 返回默认响应
        return DirectorResponse(
            action="continue"
        )
    except Exception as e:
        print(f"❌ Unexpected Error in Parser: {type(e).__name__}: {e}")
        print(f"Full Response: {response}")
        # 返回默认响应
        # 返回默认响应
        return DirectorResponse(
            action="continue"
        )


# ==================== 自动选择可用的模型 ====================

def get_available_provider() -> tuple[str, callable]:
    """获取首个可用的 Provider (优先竞速模式)"""
    
    # 0. 优先尝试竞速模式 (如果有 MiniMax + StepFun)
    # USER REQUEST: 暂时关闭竞速模式，仅使用 MiniMax
    # if os.getenv("MINIMAX_API_KEY") and os.getenv("STEPFUN_API_KEY"):
    #     return "race", call_race_vision
        
    # 1. 检查各厂商 API Key (按优先级)
    priority_list = [
        ("minimax", call_minimax_vision),
        ("stepfun", call_stepfun_vision),
        ("openai", call_openai_vision), 
        ("gemini", call_gemini_vision),
        ("qwen", call_qwen_vision),
        ("zhipu", call_zhipu_vision),
        ("baichuan", call_baichuan_vision),
        ("moonshot", call_moonshot_vision),
        ("deepseek", call_deepseek_vision)
    ]

    for provider, func in priority_list:
        env_key = MODEL_PROVIDERS[provider]["env_key"]
        if os.getenv(env_key):
            return provider, func

    return None, None


# ==================== API 端点 ====================

@app.post("/analyze_image", response_model=APIResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """
    分析图片并返回拍照建议
    支持多种模型：OpenAI, Gemini, 通义千问, 智谱GLM, 月之暗面, 阶跃星辰, 百川, DeepSeek
    支持图片格式：JPEG, WebP, PNG
    """
    import time
    start_time = time.time()
    
    try:
        # 验证 base64 图片数据
        image_data = request.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]
        try:
            decoded = base64.b64decode(image_data)
            # 记录图片大小
            image_size_kb = len(decoded) / 1024
            print(f"📷 接收图片: {image_size_kb:.1f}KB, 格式: {request.image_format}")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # 根据格式构建正确的 data URI
        mime_type = f"image/{request.image_format or 'jpeg'}"
        
        # 准备用户风格上下文
        global USER_STYLE_PROFILE
        style_context = ""
        if USER_STYLE_PROFILE:
            style_context = f"**用户偏好的摄影风格**：{USER_STYLE_PROFILE}"
            print(f"🎨 应用用户风格: {USER_STYLE_PROFILE[:20]}...")

        # 根据请求或自动选择模型
        provider_map = {
            "minimax": call_minimax_vision,
            "openai": call_openai_vision,
            "gemini": call_gemini_vision,
            "qwen": call_qwen_vision,
            "zhipu": call_zhipu_vision,
            "moonshot": call_moonshot_vision,
            "stepfun": call_stepfun_vision,
            "baichuan": call_baichuan_vision,
            "deepseek": call_deepseek_vision,
            "mcp": call_mcp_understand_image,  # MCP 协议
            "race": call_race_vision,          # 极速竞速模式
        }
        
        if request.provider and request.provider in provider_map:
            # 使用指定的模型
            provider = request.provider
            call_func = provider_map[provider]

            # 验证 API Key
            env_key = MODEL_PROVIDERS[provider]["env_key"]
            if not os.getenv(env_key):
                raise HTTPException(status_code=500, detail=f"{env_key} not configured")
        else:
            # 自动选择
            provider, call_func = get_available_provider()
            if not provider:
                raise HTTPException(
                    status_code=500,
                    detail="No API key configured. Please set one of: " + 
                           ", ".join([v["env_key"] for v in MODEL_PROVIDERS.values()])
                )
        
        # 调用 API（传递格式信息）
        # 注意：我们需要将 style_context 传递给 vision 函数
        if provider == "minimax": 
            # MiniMax 支持 style_context
            response = await call_minimax_vision(image_data, request.user_message, request.image_format, style_context)
        elif provider == "race":
             # Race 模式也支持 style_context
            response = await call_race_vision(image_data, request.user_message, request.image_format, style_context)
        else:
            # 其他模型暂未更新签名，通过 prompt 拼接方式支持
            user_msg_with_style = request.user_message or "请分析并指导"
            if style_context:
                user_msg_with_style += f"\n\n(注意：{style_context})"
            response = await call_func(image_data, user_msg_with_style, request.image_format)
        
        # 解析响应
        director_response = parse_llm_response(response, provider)
        
        # 记录处理时间
        elapsed_time = (time.time() - start_time) * 1000
        print(f"✅ 分析完成: {elapsed_time:.0f}ms, 提供商: {provider}")
        
        return APIResponse(success=True, data=director_response, provider=provider)
        
    except HTTPException:
        raise
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        print(f"❌ 分析失败: {elapsed_time:.0f}ms, 错误: {str(e)}")
        return APIResponse(success=False, error=str(e))


@app.post("/learn_style", response_model=APIResponse)
async def learn_style(request: StyleLearningRequest):
    """
    学习用户上传图片的风格（本地隐私存储）
    """
    print(f"📚 收到风格学习请求，图片数量: {len(request.images)}")
    
    try:
        if not request.images:
             raise HTTPException(status_code=400, detail="No images provided")

        # 确保目录存在
        ensure_style_dir()
        
        # 保存所有上传的图片到本地
        saved_paths = []
        for idx, img_base64 in enumerate(request.images):
            path = save_style_image(img_base64, idx)
            if path:
                saved_paths.append(path)
        
        print(f"💾 已保存 {len(saved_paths)} 张图片到本地")

        # 调用 MiniMax 分析风格
        style_description = await call_minimax_style_learning(request.images)
        
        # 更新全局风格画像
        global USER_STYLE_PROFILE
        USER_STYLE_PROFILE = style_description
        
        # 持久化保存到本地
        save_user_profile()
        
        print(f"✅ 风格学习完成: {USER_STYLE_PROFILE}")
        
        return APIResponse(success=True, style_profile=USER_STYLE_PROFILE)
    except Exception as e:
        print(f"❌ 风格学习失败: {e}")
        return APIResponse(success=False, error=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点 - 显示所有配置的模型"""
    configured = {}
    for provider, config in MODEL_PROVIDERS.items():
        configured[provider] = {
            "name": config["name"],
            "configured": bool(os.getenv(config["env_key"]))
        }
    return {"status": "healthy", "providers": configured}


# ==================== 个人风格库 API ====================

@app.get("/style_library")
async def get_style_library():
    """获取风格库文件夹和图片列表"""
    supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    folders = []
    
    if not os.path.exists(STYLE_DIR):
        return {"folders": [], "profile_exists": False}
    
    for item in os.listdir(STYLE_DIR):
        item_path = os.path.join(STYLE_DIR, item)
        if os.path.isdir(item_path):
            images = []
            for file in os.listdir(item_path):
                if os.path.splitext(file)[1].lower() in supported_formats:
                    images.append(file)
            
            if images:
                folders.append({
                    "name": item,
                    "image_count": len(images),
                    "images": sorted(images)[:12]  # 只返回前12张
                })
    
    profile_exists = os.path.exists(STYLE_PROFILE_PATH)
    
    return {
        "folders": sorted(folders, key=lambda x: x["name"]),
        "profile_exists": profile_exists
    }


class CreateFolderRequest(BaseModel):
    folder_name: str


@app.post("/create_folder")
async def create_folder(request: CreateFolderRequest):
    """创建新的风格文件夹"""
    folder_name = request.folder_name.strip()
    
    # 验证文件夹名称
    if not folder_name:
        return {"success": False, "error": "文件夹名称不能为空"}
    
    # 安全检查
    if ".." in folder_name or "/" in folder_name or "\\" in folder_name:
        return {"success": False, "error": "非法的文件夹名称"}
    
    folder_path = os.path.join(STYLE_DIR, folder_name)
    
    if os.path.exists(folder_path):
        return {"success": False, "error": "文件夹已存在"}
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        return {"success": True, "folder": folder_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/upload_to_folder")
async def upload_to_folder(folder_name: str = Form(...), files: list[UploadFile] = File(...)):
    """上传图片到指定文件夹"""
    # 安全检查
    if ".." in folder_name or "/" in folder_name or "\\" in folder_name:
        return {"success": False, "error": "非法的文件夹名称"}
    
    folder_path = os.path.join(STYLE_DIR, folder_name)
    
    # 如果文件夹不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    saved_count = 0
    for file in files:
        # 验证文件类型
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        
        # 生成唯一文件名
        ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        existing_files = os.listdir(folder_path)
        new_index = len(existing_files) + 1
        new_filename = f"{new_index}{ext}"
        
        # 确保文件名唯一
        while new_filename in existing_files:
            new_index += 1
            new_filename = f"{new_index}{ext}"
        
        file_path = os.path.join(folder_path, new_filename)
        
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_count += 1
        except Exception as e:
            print(f"保存文件失败: {e}")
    
    return {"success": True, "count": saved_count, "folder": folder_name}


@app.delete("/delete_folder/{folder_name}")
async def delete_folder(folder_name: str):
    """删除风格文件夹"""
    # 安全检查
    if ".." in folder_name or "/" in folder_name or "\\" in folder_name:
        return {"success": False, "error": "非法的文件夹名称"}
    
    folder_path = os.path.join(STYLE_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        return {"success": False, "error": "文件夹不存在"}
    
    try:
        shutil.rmtree(folder_path)
        return {"success": True, "message": f"文件夹 {folder_name} 已删除"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete("/delete_image/{folder_name}/{filename}")
async def delete_image(folder_name: str, filename: str):
    """删除风格图片"""
    # 安全检查
    if ".." in folder_name or ".." in filename or "/" in folder_name or "/" in filename:
        return {"success": False, "error": "非法的文件路径"}
    
    file_path = os.path.join(STYLE_DIR, folder_name, filename)
    
    if not os.path.exists(file_path):
        return {"success": False, "error": "图片不存在"}
    
    try:
        os.remove(file_path)
        return {"success": True, "message": f"图片 {filename} 已从 {folder_name} 删除"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/style_image/{folder}/{filename}")
async def get_style_image(folder: str, filename: str):
    """获取风格图片"""
    from fastapi.responses import FileResponse
    
    # 安全检查：防止路径遍历攻击
    if ".." in folder or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    image_path = os.path.join(STYLE_DIR, folder, filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)


@app.get("/style_folder/{folder_name}")
async def get_style_folder(folder_name: str):
    """获取文件夹中的所有图片列表"""
    supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    
    # 安全检查
    if ".." in folder_name:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    folder_path = os.path.join(STYLE_DIR, folder_name)
    
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Folder not found")
    
    images = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in supported_formats:
            images.append(file)
    
    return {
        "folder": folder_name,
        "images": sorted(images),
        "total": len(images)
    }


@app.post("/relearn_styles")
async def relearn_styles():
    """重新学习风格 - 从本地图片目录学习用户风格"""
    global USER_STYLE_PROFILE
    
    # 检查是否有图片
    if not os.path.exists(STYLE_DIR):
        return {"success": False, "error": "风格目录不存在，请先上传图片"}
    
    supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    style_categories = {}
    
    # 扫描所有风格文件夹
    for item in os.listdir(STYLE_DIR):
        item_path = os.path.join(STYLE_DIR, item)
        if os.path.isdir(item_path):
            images = []
            for file in os.listdir(item_path):
                if os.path.splitext(file)[1].lower() in supported_formats:
                    images.append(os.path.join(item_path, file))
            if images:
                style_categories[item] = images[:5]  # 每个类别最多5张
    
    if not style_categories:
        return {"success": False, "error": "未找到任何风格图片，请上传图片到 user_styles 文件夹"}
    
    # 逐个类别分析
    all_styles = {}
    errors = []
    
    for category, image_paths in style_categories.items():
        try:
            # 读取图片并转换为 base64
            images_base64 = []
            for img_path in image_paths:
                with open(img_path, 'rb') as f:
                    images_base64.append(base64.b64encode(f.read()).decode('utf-8'))
            
            # 调用 MiniMax 分析风格
            style_description = await call_minimax_style_learning(images_base64)
            all_styles[category] = style_description
            print(f"✅ 分析完成: {category}")
        except Exception as e:
            errors.append(f"{category}: {str(e)}")
            print(f"❌ 分析失败 {category}: {e}")
    
    if not all_styles:
        return {"success": False, "error": f"所有类别分析失败: {'; '.join(errors)}"}
    
    # 合并风格描述并保存
    combined_style = "\n".join([f"【{cat}】{desc}" for cat, desc in all_styles.items()])
    USER_STYLE_PROFILE = combined_style
    
    # 保存到本地文件
    try:
        profile_data = {
            "updated_at": str(os.path.getmtime(STYLE_PROFILE_PATH)) if os.path.exists(STYLE_PROFILE_PATH) else "new",
            "total_categories": len(all_styles),
            "styles": {cat: {"description": desc} for cat, desc in all_styles.items()}
        }
        with open(STYLE_PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 风格画像已保存: {STYLE_PROFILE_PATH}")
    except Exception as e:
        print(f"⚠️ 保存失败: {e}")
    
    return {
        "success": True,
        "message": f"已学习 {len(all_styles)} 个风格类别",
        "categories": list(all_styles.keys())
    }


@app.get("/providers")
async def list_providers():
    """列出支持的所有模型提供商"""
    result = []
    for provider, config in MODEL_PROVIDERS.items():
        result.append({
            "id": provider,
            "name": config["name"],
            "configured": bool(os.getenv(config["env_key"])),
            "env_key": config["env_key"],
            "mcp_enabled": config.get("mcp_enabled", False)
        })
    return {"providers": result}


@app.get("/mcp/info")
async def mcp_info():
    """获取 MCP 配置信息"""
    return {
        "enabled": bool(os.getenv("MCP_SERVER_URL")),
        "server_url": os.getenv("MCP_SERVER_URL", ""),
        "server_name": MCP_SERVER_CONFIG.name,
        "tools": [t.to_dict() for t in MCP_SERVER_CONFIG.tools]
    }


@app.post("/mcp/understand_image")
async def mcp_understand_image(request: ImageAnalysisRequest):
    """
    通过 MCP 协议调用图片理解工具
    
    使用方法：
    - 前端发送图片 base64 和用户消息
    - 后端通过 MCP 协议调用图片理解服务
    - 返回分析结果
    """
    import time
    start_time = time.time()
    
    try:
        # 验证并提取 base64 数据
        image_data = request.image_base64
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        # 调用 MCP 图片理解工具
        response = await call_mcp_understand_image(image_data, request.user_message)
        
        elapsed_time = (time.time() - start_time) * 1000
        print(f"✅ MCP 图片理解完成: {elapsed_time:.0f}ms")
        
        # 解析响应
        director_response = parse_llm_response(response, "mcp")
        
        return APIResponse(success=True, data=director_response, provider="mcp")
        
    except Exception as e:
        elapsed_time = (time.time() - start_time) * 1000
        print(f"❌ MCP 图片理解失败: {elapsed_time:.0f}ms, 错误: {str(e)}")
        return APIResponse(success=False, error=str(e))


@app.get("/pose_library")
async def get_pose_library():
    """获取支持的姿势库"""
    return {
        "poses": [
            {"id": "standing_casual", "name": "自然站立", "description": "以此为基础的自然站立姿态"},
            {"id": "leaning_wall", "name": "倚靠墙壁", "description": "靠在墙壁或柱子上的放松姿态"},
            {"id": "sitting_coffee", "name": "坐着喝东西", "description": "坐着拿饮料的休闲姿态"},
            {"id": "walking_away", "name": "背影行走", "description": "行走中的背影照"},
            {"id": "peace_sign", "name": "比耶", "description": "举手比耶的活泼姿势"}
        ]
    }


# ==================== 模块二：姿态匹配 API ====================

from pose_matcher import (
    Landmark,
    landmarks_from_dict,
    calculate_pose_similarity,
    get_feedback_instruction,
    get_detailed_analysis,
    TARGET_POSES
)


class PoseMatchRequest(BaseModel):
    """姿态匹配请求"""
    landmarks: list = Field(..., description="MediaPipe Pose 33个关键点数组")
    target_pose_id: str = Field(..., description="目标姿势ID")


class PoseMatchResponse(BaseModel):
    """姿态匹配响应"""
    score: float = Field(..., description="匹配分数 0-100")
    is_match: bool = Field(..., description="是否匹配成功（>=70分）")
    feedback: Optional[str] = Field(None, description="反馈指令")
    breakdown: Optional[dict] = Field(None, description="各部位分数细节")


@app.post("/match_pose", response_model=PoseMatchResponse)
async def match_pose(request: PoseMatchRequest):
    """实时姿态匹配接口（无需调用LLM，速度极快）"""
    try:
        if request.target_pose_id not in TARGET_POSES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown pose ID: {request.target_pose_id}. Available: {list(TARGET_POSES.keys())}"
            )
        
        landmarks = landmarks_from_dict(request.landmarks)
        analysis = get_detailed_analysis(landmarks, request.target_pose_id)
        
        return PoseMatchResponse(
            score=analysis["overall_score"],
            is_match=analysis["is_match"],
            feedback=analysis["feedback"],
            breakdown=analysis["breakdown"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose matching error: {str(e)}")


# ==================== 语音合成 (TTS) API ====================

class TTSRequest(BaseModel):
    """TTS 请求模型"""
    text: str = Field(..., description="要转换为语音的文本")
    voice: Optional[str] = Field("cixingnansheng", description="音色ID")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    阶跃星辰 TTS 语音合成
    返回 base64 编码的音频数据
    """
    api_key = os.getenv("STEPFUN_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="STEPFUN_API_KEY not configured")
    
    # 限制文本长度
    text = request.text[:500]  # 最多500字符
    
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    @retry(
        retry=retry_if_exception_type(HTTPException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def call_tts_api():
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.stepfun.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "step-tts-mini",
                    "input": text,
                    "voice": request.voice if request.voice in ["cixingnansheng", "tianmeiyujie", "zhixingnvsheng", "wenrounvsheng", "yuanqishaonv", "yangguangnanhai"] else "cixingnansheng",
                    "response_format": "mp3",
                    "language": "zh"
                }
            )
            
            if response.status_code == 429:
                print("TTS 429 限流，正在重试...")
                raise HTTPException(status_code=429, detail="Rate limited")
            
            if response.status_code != 200:
                print(f"TTS API 错误: {response.status_code}, {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"TTS API error: {response.text}")
                
            return response.content

    try:
        audio_content = await call_tts_api()
        
        # 返回 base64 编码的音频
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        audio_size_kb = len(audio_content) / 1024
        print(f"🔊 TTS 生成: {len(text)}字 -> {audio_size_kb:.1f}KB")
        
        return {
            "success": True,
            "audio_base64": audio_base64,
            "format": "mp3",
            "size_kb": round(audio_size_kb, 1)
        }
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS request timeout")
    except Exception as e:
        print(f"TTS 错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.get("/tts/voices")
async def list_tts_voices():
    """获取可用的 TTS 音色列表"""
    return {
        "voices": [
            {"id": "cixingnansheng", "name": "磁性男声", "gender": "male", "description": "温暖磁性的男性声音"},
            {"id": "tianmeiyujie", "name": "甜美御姐", "gender": "female", "description": "甜美成熟的女性声音"},
            {"id": "zhixingnvsheng", "name": "知性女声", "gender": "female", "description": "知性优雅的女性声音"},
            {"id": "wenrounnvsheng", "name": "温柔女声", "gender": "female", "description": "温柔亲切的女性声音"},
            {"id": "yuanqishaonv", "name": "元气少女", "gender": "female", "description": "活泼元气的少女声音"},
            {"id": "yangguangnanhai", "name": "阳光男孩", "gender": "male", "description": "阳光开朗的男孩声音"},
        ],
        "default": "cixingnansheng"
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    # 初始化本地存储目录
    ensure_style_dir()
    
    # 加载用户风格画像
    load_user_profile()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
