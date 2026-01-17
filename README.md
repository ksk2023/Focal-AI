---
title: FocalAI - AI Photography Assistant
emoji: camera
colorFrom: blue
colorTo: purple
sdk: docker
domain: multi-modal
tags:
  - ai-photography
  - pose-detection
  - mediapipe
  - fastapi
license: Apache License 2.0
---

# FocalAI - AI Photography Assistant

AI拍照助手 - 帮助不会拍照的人群拍出美美的照片

## Features

- AI Director: 智能场景分析与构图建议
- Pose Guidance: 实时姿势引导与 AR 叠加
- Style Learning: 学习用户喜好的拍照风格
- Voice Feedback: 语音反馈指导

## Architecture

```
Frontend (Mobile UI)  ──┐
                        ├──> Nginx (7860) ──> Backend (FastAPI 8000)
                        │
Static HTML/JS    <─────┘
```

## Environment Variables

在魔搭创空间的「设置」中配置以下环境变量（至少配置一个 AI 模型的 API Key）：

| 变量名 | 说明 |
|--------|------|
| `MINIMAX_API_KEY` | MiniMax API Key (推荐) |
| `MINIMAX_GROUP_ID` | MiniMax Group ID |
| `QWEN_API_KEY` | 通义千问 API Key |
| `ZHIPU_API_KEY` | 智谱 GLM API Key |
| `OPENAI_API_KEY` | OpenAI API Key |
| `GEMINI_API_KEY` | Google Gemini API Key |

## Local Development

```bash
# Clone
git clone https://www.modelscope.cn/studios/kskyqsl/FocalAI.git
cd FocalAI

# Setup
cp .env.example .env
# Edit .env and add your API keys

# Run backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Run frontend (in another terminal)
cd mobile
python -m http.server 3001

# Access: http://localhost:3001?api=http://localhost:8000
```

## Clone with HTTP

```bash
git clone https://www.modelscope.cn/studios/kskyqsl/FocalAI.git
```
