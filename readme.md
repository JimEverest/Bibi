# Bibi — 离线语音输入工具（本地定制版）

**Bibi** 是一款完全离线运行的桌面语音输入工具：按住/按下快捷键说话，识别结果自动键入当前光标位置。基于 FunASR（SenseVoice）本地 ONNX 推理，**不上传任何音频，断网可用**，纯 CPU 运行，无需显卡。

本版本是针对个人工作环境深度定制与功能增强的版本，面向内网/离线环境部署。

---

## 功能特性

### 核心
- **100% 离线识别**：FunASR + SenseVoice 本地 ONNX 推理，音频不出电脑
- **中英混合识别**：旗舰级模型，中英穿插口述同样精准
- **全局热键听写**：任何应用、任何文本框内直接语音输入（默认 `F2` 开/关）
- **按住说话（PTT）**：长按快捷键录音、松开即停止出字（默认 `Win+Ctrl+Alt`）
- **文本注入多策略**：SendInput / 剪贴板 / Unicode 自动降级（`auto`）

### 智能润色（可选）
- **LLM 二次润色**：纠同音错字、去口癖、合并重复、整合自我修正，保留原语气
- **双协议支持**：OpenAI（Chat Completions / Responses）与 Anthropic（Messages API）
- **思考模式开关**：DeepSeek V4 等默认 thinking=high 的模型可一键关闭，润色从十几秒降至约 1 秒
- **Prompt 可配置**：内置同音纠错增强 Prompt，支持 `{TEXT}` / `{VOCAB}` 占位符
- **领域词汇注入**：配置词汇表 + terms.txt 词典自动注入，专有名词一次就对
- **失败自动降级**：LLM 超时/出错时无缝回退 ASR 原文，不阻塞输入

### 本地词典
- **terms.txt 替换词典**：人名、术语、缩写自定义替换，识别后即时生效

### 工程化
- **系统托盘**：状态变色图标（待机灰蓝/录音红）、右键菜单操作、版本号显示
- **录音指示浮窗**：右下角无边框小窗，录音计时 / 处理中提示
- **设置界面**：GUI 修改全部配置，支持 LLM 连接一键测试
- **录音持久化开关**：默认不保存 WAV；调试需要时可在设置中开启
- **一键清理**：托盘设置内一键清除全部日志与录音文件（后台执行不卡界面）
- **单实例锁**：端口锁防止多开；崩溃自动恢复提示

## 快速开始

### 环境依赖
- Windows 10/11
- Python 3.12+（建议 venv 隔离）
- 麦克风权限

### 安装与运行

```bat
:: 1. 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate

:: 2. 安装依赖
pip install -r requirements.txt

:: 3. 运行（首次运行自动下载约 500MB 模型到 models/）
python main.py
```

启动后托盘出现麦克风图标，按 `F2`（或长按 `Win+Ctrl+Alt`）即可语音输入。

### 离线部署（内网环境）

使用 `vocotype_offline_bundle/` 离线包（含模型 + wheels 依赖 + 一键安装脚本），拷贝到目标机后运行 `内网离线安装.bat` 即可，全程无需联网。详见 `内网部署手册.md`。

## 配置说明

配置文件为项目根目录的 `vocotype_config.json`，启动自动加载；也可通过托盘菜单「设置…」在 GUI 中修改。主要配置段：

| 配置段 | 关键项 | 说明 |
|---|---|---|
| `hotkeys` | `toggle` / `push_to_talk` | 开关式热键 / 按住说话热键（`none` 禁用，重启生效） |
| `audio` | `gain` / `max_session_bytes` / `save_recordings` | 麦克风增益、单次录音上限、WAV 持久化（默认关） |
| `output` | `method` | 文本输出方式：`auto` / `type` / `clipboard` / `unicode` |
| `llm` | `enabled` / `schema` / `endpoint` / `api_key` / `model` | LLM 润色开关与连接信息 |
| `llm` | `style` | OpenAI 协议端点风格：`chat`（/chat/completions）或 `responses`（/v1/responses）；**schema=anthropic 时使用 Messages API，此项忽略** |
| `llm` | `disable_thinking` | 关闭模型思考模式（默认 true，强烈建议保持） |
| `llm` | `vocabulary` / `prompt` | 领域词汇（逗号分隔）/ 自定义润色 Prompt |
| `ui` | `show_indicator` | 录音指示浮窗开关 |

### LLM 润色配置示例（DeepSeek）

```json
{
  "llm": {
    "enabled": true,
    "schema": "openai",
    "style": "chat",
    "endpoint": "https://api.deepseek.com/v1/chat/completions",
    "api_key": "sk-xxxx",
    "model": "deepseek-v4-flash",
    "temperature": 0.3,
    "timeout_seconds": 15,
    "disable_thinking": true
  }
}
```

若使用 Anthropic 兼容端点（如 DeepSeek 的 `/anthropic/v1/messages`），把 `schema` 改为 `"anthropic"` 即可，无需设置 `style`。

## 快捷键

| 默认热键 | 功能 |
|---|---|
| `F2` | 开始/停止录音（开关模式） |
| `Win+Ctrl+Alt`（按住） | 按住说话，松开自动停止 |
| 托盘菜单 | 录音开关 / LLM 润色开关 / 设置 / 日志目录 / 退出 |

所有热键均可在设置界面修改，重启程序生效。

## 模型说明

### 各功能模块与所用模型

| 功能模块 | 引擎/服务 | 模型 | 说明 |
|---|---|---|---|
| 语音识别（核心） | 本地离线 FunASR-ONNX | `iic/SenseVoiceSmall`（默认） | 中英混合识别，自带标点与 ITN，自动剥离情绪标签 |
| 语音识别（回退） | 本地离线 FunASR-ONNX | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx` | 旧引擎，中文为主；名字带 "large" 是架构代号，**不是**新引擎 |
| 端点检测 VAD | 本地离线 FunASR-ONNX | `iic/speech_fsmn_vad_zh-cn-16k-common-onnx` | 仅 534KB；按 `asr.use_vad` 开关加载（默认关） |
| 标点恢复 | 本地离线 FunASR-ONNX | `iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx` | 仅 paraformer 引擎需要（SenseVoice 自带标点） |
| 云端识别（可选） | 火山引擎 BigASR 流式 | 云端大模型，无需下载 | `backend: "volcengine"` 时启用，需 App Key/Access Key；音频会上传 |
| LLM 润色（可选） | 外部 LLM API | 任意 OpenAI/Anthropic 兼容模型 | 已实测 DeepSeek（`deepseek-v4-flash`）双协议；建议关闭 thinking 提速 |
| 文本注入 / 热键 / 词典 / 托盘 | — | 无模型 | 纯本地逻辑 |

### 本地模型存放位置

模型存放于 ModelScope 标准缓存目录（程序按此路径查找，**目录结构不可变**）：

```
%USERPROFILE%\.cache\modelscope\hub\models\iic\
├── SenseVoiceSmall\                                          # 默认引擎，必需（约 232MB，用量化版 model_quant.onnx）
│   ├── model_quant.onnx          # 量化识别模型
│   ├── chn_jpn_yue_eng_ko_spectok.bpe.model
│   ├── am.mvn
│   └── config.yaml
├── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx\   # 可选回退（228MB）
├── speech_fsmn_vad_zh-cn-16k-common-onnx\                    # 可选 VAD（534KB）
└── punc_ct-transformer_zh-cn-common-vocab272727-onnx\        # 可选标点（274MB，仅 paraformer 用）
```

**下载方式**（二选一）：
- **自动**：联网环境下首次运行 `python main.py`，程序自动从 ModelScope 下载（约 500MB，含全部模型）
- **手动**：`python -c "from app.download_models import main; main()"` 或从 ModelScope 页面直接下载后放入上述目录：
  - https://modelscope.cn/models/iic/SenseVoiceSmall
  - https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx
  - https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx
  - https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx

**最省方案**：只放 `SenseVoiceSmall\`（232MB）即可跑默认引擎。离线内网部署请直接用 `vocotype_offline_bundle/`（已内置全部模型），详见 `内网部署手册.md`。

### 引擎切换

环境变量 `ASR_ENGINE` 控制（`启动语音输入.bat` 已默认设置）：

```bat
set ASR_ENGINE=sensevoice   :: 默认，中英混说更准
set ASR_ENGINE=paraformer   :: 回退，中文为主，支持热词（需另配标点模型）
```

> 别被名字迷惑：**Paraformer-"large" 是旧引擎**；新默认引擎的模型反而叫 **SenseVoice-"Small"**——"large/small" 是架构代号，不代表效果。

### 实测结论与其他模型方案

| 方案 | 结论 |
|---|---|
| **SenseVoiceSmall**（当前默认） | ✅ 实测最优：中英混说显著优于 Paraformer，英文长句几乎全对；加载 2.6s、13s 音频推理约 1s |
| Paraformer-large（ONNX） | 旧默认，中文准但英文弱（长句碎成单词）；保留作回退 |
| SenseVoiceSmall FP32（model.onnx） | 未采用：比量化版大且速度无优势，量化版精度损失可忽略 |
| FunASR 在线/流式模型 | 未采用：本工具为"说完一句转一句"的非流式场景，非流式模型更准 |
| whisper.cpp / faster-whisper | 备选未用：英文强但中文场景体积/速度/易用性不如 SenseVoice；且 CPU 上 small 以上模型偏慢 |
| 火山引擎 BigASR（云端） | 已接入可选：识别质量旗舰级、无需下载模型，但需联网且音频出本机（隐私敏感场景不适用） |

## 目录结构

```
bibi/
├── main.py                  # 入口：热键注册、录音循环、UIHub 主线程事件泵
├── VERSION                  # 版本号（托盘/设置窗口显示）
├── terms.txt                # 本地替换词典
├── vocotype_config.json     # 配置文件（启动自动加载）
├── app/
│   ├── appinfo.py           # 应用名称与版本号（唯一出处）
│   ├── funasr_server.py     # FunASR 模型服务器（模型常驻内存，stdin/stdout 通信）
│   ├── funasr_config.py     # 模型名称/版本/引擎统一配置
│   ├── download_models.py   # 模型下载（ModelScope snapshot_download）
│   ├── volcengine_asr.py    # 火山引擎云端流式识别后端（可选）
│   ├── audio_capture.py     # 麦克风采集与增益
│   ├── transcribe.py        # 转写编排（录音→ASR→词典→LLM→注入）
│   ├── llm_polish.py        # LLM 润色（OpenAI/Anthropic 双协议）
│   ├── dictionary.py        # 词典替换 + 词汇提示
│   ├── hotkeys.py           # 热键管理（开关式 + 按住说话 PTT）
│   ├── indicator.py         # 录音指示浮窗
│   ├── tray.py              # 系统托盘
│   ├── settings_dialog.py   # 设置界面
│   ├── ui_hub.py            # UIHub 单例（唯一 Tk root + 跨线程队列派发）
│   ├── singleton.py         # 单实例端口锁
│   ├── typer.py             # 文本注入（SendInput/剪贴板/Unicode）
│   └── config.py            # 配置加载与默认值
└── models/                  # 本地模型文件（首次运行自动下载，或见「模型说明」一节）
```

## 常见问题

**Q: 数据安全吗？**
A: 安全。语音识别完全本地离线，音频不上传。仅在显式开启 LLM 润色时，转写文本才会发送到你配置的 LLM 服务商。

**Q: LLM 润色很慢（十几秒）？**
A: 新模型（DeepSeek V4 等）默认开启思考模式。在设置中勾选「关闭模型思考模式」（默认已勾选），润色可降至约 1 秒。

**Q: 换了快捷键不生效？**
A: 热键修改需重启程序（设置界面有提示）。

**Q: 提示「已有实例在运行」？**
A: 程序为单实例设计。在任务管理器中结束旧的 `python.exe` 后再启动。

**Q: 日志/录音文件堆积？**
A: WAV 持久化默认关闭；如曾开启，可在设置中点击「一键清理日志与录音文件」。

## 致谢

Bibi 基于以下优秀的开源项目构建（rebrand 自 vocotype-cli）：

- **[FunASR](https://github.com/modelscope/FunASR)** — 阿里达摩院开源语音识别框架，提供离线识别能力
- **[vocotype-cli](https://github.com/233stone/vocotype-cli)** — 原始项目
- **[QuQu](https://github.com/yan5xu/ququ)** — 润色 Prompt 设计参考
