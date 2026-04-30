# 作业8：Deepfake 假视频制作

详细代码可见代码附件，或看我的github仓库：https://github.com/moonlight-liu/Content_Security_Experiment

## 制作流程

### 1. 环境配置
使用 conda `cv3` 环境（PyTorch + CUDA），安装 `edge-tts` 用于中文语音合成，克隆 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) 仓库进行唇形同步，并下载 `wav2lip_gan.pt` 预训练模型。

### 2. 中文语音生成 — `task2_generate_audio.py`
使用 edge-tts 的「云健」中文男声，生成约19秒的语音，内容为特朗普用中文夸奖刘翔。

### 3. 视频预处理
原始视频中特朗普人脸区域较小（约90×90像素），先使用 FFmpeg 裁切放大脸部区域至 640×480，使 Wav2Lip 能有效识别嘴部。

### 4. 唇形同步 — `task3_lipsync.py`
调用 Wav2Lip 的 `inference.py`，将特朗普视频画面与中文语音进行唇形同步，生成嘴部随语音变化的视频。同时打了 PyTorch 2.11 兼容性补丁。

### 5. 视频压缩 — `task4_compress.py`
使用 FFmpeg 将 Wav2Lip 输出的 AVI 文件转码为 H.264 + AAC 编码的 MP4 格式，最终大小约 1.5MB，满足 ≤10MB 要求。

### 6. 验证 — `task5_verify.py`
检查最终视频的文件大小、时长、分辨率和编码格式。

## 依赖
- `edge-tts` — 中文语音合成
- `Wav2Lip` — 唇形同步
- `FFmpeg` — 视频处理
- `PyTorch` + CUDA — GPU 加速推理

## 最终输出
`final_result.mp4` — 640×480, 19秒, 1.5MB, MP4格式
