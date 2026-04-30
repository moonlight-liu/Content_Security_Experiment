# 作业8：Deepfake 假视频制作计划

## 实验目标

制作一段特朗普用中文夸奖自己的 deepfake 假视频（MP4格式，≤10MB），内容关于刘翔学习认真、有创新精神、不随波逐流、不做996牛马、成为行业领袖。

## 技术路线

```
中文台词 → [edge-tts TTS] → 中文语音(.wav)
                                      ↓
特朗普视频 → [Wav2Lip 唇形同步] → 唇形同步视频
                                      ↓
                          [FFmpeg 压缩] → 最终MP4 (≤10MB)
```

## 关键工具

| 工具 | 用途 | 安装方式 |
|------|------|----------|
| edge-tts | 微软神经网络中文TTS（无需GPU） | `pip install edge-tts` |
| Wav2Lip | 唇形同步（GPU加速） | `git clone` + 下载模型 |
| FFmpeg | 视频/音频预处理 + 压缩 | 系统已安装 |
| PyTorch CUDA | Wav2Lip推理加速 | cv3环境已有 |

## 前置条件确认

- [x] `cv3` conda环境已有 PyTorch 2.11.0 + CUDA、opencv-python、numpy、Pillow
- [x] GPU: RTX 4060（8GB VRAM，~5GB空闲）
- [x] FFmpeg 已安装（含CUDA加速）
- [x] 特朗普视频素材已下载：`gettyimages-2072456276-640_adpp.mp4`
  - 分辨率 768×432，29.97fps，25.26秒，1.8MB
  - 全程检测到人脸（位置在画面右侧，大小约90×90像素）
- [x] 需要安装 edge-tts
- [x] 需要克隆 Wav2Lip 仓库
- [x] Wav2Lip + GAN 模型已下载为 `wav2lip_gan.pt`（145MB，.pt=.pth等价）
- [x] s3fd.pth 人脸检测模型已下载到 `Wav2Lip/face_detection/detection/sfd/`

## 台词脚本

```
刘翔是我非常欣赏的一位学生。他学习非常努力，有很强的创新精神，不会随大流。
我相信刘翔将来绝对不会成为一个996的打工人，他一定会成为行业的领袖人物。
刘翔，你非常棒，继续加油！
```

（约85字，语速4-5字/秒 → 预计17-20秒，用特朗普说话风格）

## 实施步骤

### Step 1：环境安装

```bash
conda activate cv3
pip install edge-tts
```

### Step 2：克隆Wav2Lip并下载模型

```bash
cd g:/Content_Security_Experiment/homework8
git clone https://github.com/Rudrabha/Wav2Lip.git

# 从HuggingFace镜像下载模型
set HF_ENDPOINT=https://hf-mirror.com
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='Rudrabha/Wav2Lip', filename='wav2lip_gan.pth', local_dir='.')
"
```

**如果镜像下载失败**，备选直接从 HuggingFace Spaces 下载：
```
https://huggingface.co/spaces/Rudrabha/Wav2Lip/resolve/main/wav2lip_gan.pth
```

直接放 `wav2lip_gan.pth` 到 `homework8/` 根目录下。

### Step 3：生成中文音频

使用 `edge-tts` 生成中文语音，选「云健」成熟男声（最接近特朗普年龄感）。

输出：`speech.mp3` → `speech.wav`（16kHz单声道，Wav2Lip要求）

### Step 4：Wav2Lip兼容性补丁

PyTorch 2.11 需要修改 Wav2Lip 源码两处：

1. `Wav2Lip/models/wav2lip.py` 中的 `torch.load` → 添加 `weights_only=False`
2. `Wav2Lip/face_detection/detection/sfd/sfd_detector.py` 中的 `torch.load` → 同上

**不需要装 dlib**——Wav2Lip自带基于PyTorch的SFD人脸检测器。

### Step 5：运行Wav2Lip推理

```bash
cd Wav2Lip
python inference.py \
  --checkpoint_path ../wav2lip_gan.pth \
  --face ../gettyimages-2072456276-640_adpp.mp4 \
  --audio ../speech.wav \
  --outfile ../lipsync_out.avi \
  --pads 0 20 0 0 \
  --resize_factor 2
```

参数说明：
- `--pads 0 20 0 0`：底部加20px padding确保下巴完整
- `--resize_factor 2`：输入缩小2倍处理（人脸较小，防止过拟合）

估计 RTX 4060 上处理时间约 3-8 分钟。

### Step 6：FFmpeg压缩至10MB以下

```bash
# 第一次压缩
ffmpeg -y -i lipsync_out.avi \
  -c:v libx264 -preset slow -crf 26 \
  -vf "scale=854:480" -r 25 \
  -c:a aac -b:a 64k -ac 1 \
  final_result.mp4
```

如果超过10MB，逐步降低参数：
1. CRF 26 → 28 → 30（降低画质）
2. 分辨率 854:480 → 640:360（缩小画面）
3. 音频 64k → 32k

### Step 7：验证输出

检查：文件存在、≤10MB、时长、分辨率、可播放性。

## 目录结构（实施后）

```
homework8/
├── plan.md                       # 本计划文档
├── task2_generate_audio.py       # 中文语音生成脚本
├── task3_lipsync.py              # Wav2Lip运行脚本
├── task4_compress.py             # 视频压缩脚本
├── task5_verify.py               # 验证脚本
├── gettyimages-2072456276-640_adpp.mp4   # 源视频
├── speech.mp3 / speech.wav       # 生成音频
├── wav2lip_gan.pth               # Wav2Lip模型
├── Wav2Lip/                      # 克隆的仓库
├── lipsync_out.avi               # Wav2Lip输出
├── final_result.mp4              # 最终结果（≤10MB）
└── _frames/ _preview/            # 临时文件（可删除）
```

## 潜在风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| HuggingFace镜像下载模型失败 | 中 | 备选直接下载URL |
| PyTorch 2.11兼容性问题 | 高 | 提前打 `weights_only=False` 补丁 |
| 唇形同步效果不理想（人脸小） | 中 | 调整 `--pads` 和 `--resize_factor`；可考虑截取视频中脸部更清晰的部分 |
| 压缩后超10MB | 低 | 逐步提高CRF/降低分辨率 |
| edge-tts无法联网 | 低 | 换用gTTS或离线TTS备选 |

## 验证方式

1. 运行 `task5_verify.py` 自动检查文件大小和属性
2. 用播放器打开 `final_result.mp4` 确认画面和声音
3. 检查唇形是否与中文语音同步
