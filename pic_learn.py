#!/usr/bin/env python3
"""
pic_learn.py - 个人风格学习工具

读取 user_styles/images/ 目录下的所有图片，
调用 AI 模型分析风格特征，并保存到 profile.json。

使用方法：
    python pic_learn.py

依赖：
    pip install httpx python-dotenv
"""

import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime

# 尝试导入必要库
try:
    import httpx
    from dotenv import load_dotenv
except ImportError:
    print("❌ 缺少依赖，请先安装：pip install httpx python-dotenv")
    sys.exit(1)

# 加载环境变量
load_dotenv()

# ==================== 配置 ====================
STYLE_DIR = Path(__file__).parent / "user_styles"
STYLE_IMAGES_DIR = STYLE_DIR / "images"
STYLE_PROFILE_PATH = STYLE_DIR / "profile.json"

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")

STYLE_LEARNING_PROMPT = """你是一位资深的摄影风格分析专家。请对这组参考图片进行深入、全面的风格分析。

## 分析维度（请逐一详细描述）

### 1. 构图特征
- 构图类型（中心构图/三分法/对角线/框架式/对称式/引导线等）
- 画面填充度（极简留白/适中/饱满紧凑）
- 主体位置偏好（居中/偏左/偏右/黄金分割点）
- 前景/中景/背景的层次感

### 2. 色彩与调色
- 整体色温倾向（暖色调/冷色调/中性）
- 饱和度偏好（高饱和/低饱和/自然）
- 对比度风格（高对比/柔和/电影感）
- 常用滤镜风格（复古胶片/清新日系/欧美杂志/自然无滤镜等）
- 主要色彩搭配

### 3. 光线运用
- 光线类型（自然光/人造光/混合光）
- 光线方向（顺光/侧光/逆光/顶光）
- 光影对比（强烈阴影/柔和漫射/高光控制）
- 拍摄时段偏好（黄金时段/蓝调时刻/正午/夜间）

### 4. 场景与环境
- 常见拍摄场景（室内/户外/城市/自然/咖啡馆等）
- 背景偏好（简洁干净/丰富有层次/虚化模糊）
- 环境氛围（都市感/文艺感/自然感/复古感）

### 5. 人物表现（如适用）
- 姿态风格（自然随性/端庄优雅/活泼动感/酷感）
- 表情偏好（微笑/严肃/自然/侧脸/回眸）
- 与环境的互动方式
- 服装风格倾向

### 6. 技术特点
- 景深偏好（大光圈浅景深/全景深）
- 焦段偏好（广角/标准/中长焦/特写）
- 清晰度与锐度

## 输出要求
请用 150-250 字详细描述这类照片的整体风格特征和拍摄习惯。
描述应该具体、精准，避免泛泛而谈。
使用专业但易懂的摄影术语。
只输出风格描述，不要其他内容。"""


def get_image_files():
    """
    递归获取 user_styles 目录下所有图片
    返回: dict[文件夹名 -> 图片路径列表]
    """
    supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    style_categories = {}
    
    if not STYLE_DIR.exists():
        print(f"⚠️ 风格目录不存在: {STYLE_DIR}")
        return style_categories
    
    # 遍历 user_styles 下的所有子目录
    for item in STYLE_DIR.iterdir():
        if item.is_dir():
            category_name = item.name
            images = []
            
            # 收集该目录下的所有图片
            for file in item.iterdir():
                if file.is_file() and file.suffix.lower() in supported_formats:
                    images.append(file)
            
            # 也检查子目录（递归一层）
            for subdir in item.iterdir():
                if subdir.is_dir():
                    for file in subdir.iterdir():
                        if file.is_file() and file.suffix.lower() in supported_formats:
                            images.append(file)
            
            if images:
                style_categories[category_name] = sorted(images)
    
    # 也检查根目录下直接放置的图片
    root_images = []
    for file in STYLE_DIR.iterdir():
        if file.is_file() and file.suffix.lower() in supported_formats:
            root_images.append(file)
    
    if root_images:
        style_categories["默认风格"] = sorted(root_images)
    
    return style_categories


def image_to_base64(filepath: Path) -> str:
    """将图片转换为 base64"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def call_minimax_style_learning(images_base64: list) -> str:
    """调用 MiniMax API 分析风格"""
    if not MINIMAX_API_KEY:
        raise ValueError("未配置 MINIMAX_API_KEY，请在 .env 文件中设置")
    
    # 构建消息内容
    content = [{"type": "text", "text": STYLE_LEARNING_PROMPT}]
    
    for idx, img_b64 in enumerate(images_base64[:5]):  # 最多5张
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    
    payload = {
        "model": "abab6.5s-chat",
        "messages": [
            {"role": "user", "content": content}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    url = f"https://api.minimax.chat/v1/text/chatcompletion_v2"
    if MINIMAX_GROUP_ID:
        url += f"?GroupId={MINIMAX_GROUP_ID}"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API 错误: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # 提取回复内容
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"无法解析响应: {result}")


def save_profile(style_descriptions: dict):
    """保存风格画像到本地（结构化格式）"""
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "updated_at": datetime.now().isoformat(),
        "total_categories": len(style_descriptions),
        "styles": {}
    }
    
    # 将每个类别单独存储
    for category, description in style_descriptions.items():
        data["styles"][category] = {
            "description": description,
            "image_count": "分析时使用最多5张"
        }
    
    with open(STYLE_PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 风格画像已保存到: {STYLE_PROFILE_PATH}")


async def main():
    print("=" * 50)
    print("🎨 个人风格学习工具")
    print("=" * 50)
    
    # 检查图片分类
    style_categories = get_image_files()
    if not style_categories:
        print(f"❌ 没有找到图片，请将图片放入 {STYLE_DIR} 下的子文件夹中")
        print(f"   文件夹名称将作为风格类别描述")
        return
    
    total_images = sum(len(imgs) for imgs in style_categories.values())
    print(f"📸 找到 {len(style_categories)} 个风格类别，共 {total_images} 张图片:\n")
    
    for category, images in style_categories.items():
        print(f"   📁 {category}/ ({len(images)} 张)")
        for img in images[:3]:  # 只显示前3张
            print(f"      - {img.name}")
        if len(images) > 3:
            print(f"      ... 还有 {len(images) - 3} 张")
    
    # 每个类别分别分析
    all_styles = {}
    
    for category, images in style_categories.items():
        print(f"\n🤖 正在分析类别: {category}...")
        
        # 转换为 base64 (每个类别最多5张)
        images_base64 = []
        for img in images[:5]:
            images_base64.append(image_to_base64(img))
        
        try:
            style_description = await call_minimax_style_learning(images_base64)
            all_styles[category] = style_description
            print(f"   ✅ 分析完成")
        except Exception as e:
            print(f"   ❌ 分析失败: {e}")
    
    if not all_styles:
        print("\n❌ 所有类别分析失败")
        return
    
    # 显示结果
    print(f"\n📝 综合风格分析结果:")
    print("-" * 40)
    for category, description in all_styles.items():
        print(f"\n【{category}】")
        print(description)
    print("-" * 40)
    
    # 保存结果（传入字典）
    save_profile(all_styles)
    
    print("\n✅ 学习完成！下次启动后端时，AI 会自动加载此风格偏好。")


if __name__ == "__main__":
    asyncio.run(main())
