# ZONOS2 TTS ComfyUI

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/Saganaki22/Zonos2_TTS-ComfyUI)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-2d7dd2)](https://github.com/comfyanonymous/ComfyUI)
[![Upstream](https://img.shields.io/badge/Upstream-Zyphra%2FZONOS2-111111)](https://github.com/Zyphra/ZONOS2)
[![Official Model](https://img.shields.io/badge/Hugging_Face-Zyphra%2FZONOS2-ffd21e)](https://huggingface.co/Zyphra/ZONOS2)
[![Native BF16 Model](https://img.shields.io/badge/Hugging_Face-drbaph%2FZONOS2--BF16-ffd21e)](https://huggingface.co/drbaph/ZONOS2-BF16)
[![Model License](https://img.shields.io/badge/Model_License-Apache--2.0-green)](https://huggingface.co/Zyphra/ZONOS2)

[English README](README.md)

这是面向 [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2) 的原生 ComfyUI 节点，支持文本转语音、仅音频的声音克隆、SDPA 与 FlashAttention 推理、原生进度显示，以及 ComfyUI/AIMDO 显存管理。

ZONOS2 是 Zyphra 最新的文本转语音模型，使用超过 600 万小时的多语言、多样化语音进行训练。它依靠 MoE 架构以较低延迟提供可与顶级 TTS 服务相当甚至更高的表现力和质量，并擅长高保真、自然的声音克隆。

推理时，模型使用 NeMo TN 规范化的 UTF-8 字节和 ECAPA-TDNN 说话人嵌入，通过 MoE 主干生成 DAC token。推理流程如下。

![ZONOS2 推理流程](https://huggingface.co/Zyphra/ZONOS2/resolve/main/assets/zonos2_arlooop_animated.gif)

> [!WARNING]
> 只能克隆您拥有或已获得明确授权的声音。严禁将本项目用于恶意冒充、欺诈、误导、骚扰、滥用、规避同意或任何蓄意造成伤害的行为。

**主要功能**

- Windows 和 Linux 上的原生进程内 PyTorch 实现。
- 标准 ComfyUI `AUDIO` 输入与 44.1 kHz `AUDIO` 输出。
- 仅通过音频完成零样本声音克隆，无需参考文本。
- 提供全部已发布的 ZONOS2 语速与音质条件桶。
- 支持 `auto`、`bf16` 和 `fp16` 运行时数据类型。
- 自动选择 FlashAttention，并在不可用时回退到 PyTorch SDPA。
- 原生 ComfyUI 进度条与 CLI 内联 `tqdm` 进度。
- 将真实张量注册到 ComfyUI 模型管理与 AIMDO。
- 模型、数据类型或注意力设置改变时进行完整卸载和重新加载。

<details>
<summary><strong>安装</strong></summary>

### ComfyUI Manager

节点被 ComfyUI Manager 收录后，可搜索 **ZONOS2 TTS** 并直接安装。

当前也可以通过 Manager 立即安装：

1. 打开 **Manager**。
2. 选择 **Install via Git URL**。
3. 输入 `https://github.com/Saganaki22/Zonos2_TTS-ComfyUI`。
4. 重启 ComfyUI。

### 手动安装

在 `ComfyUI/custom_nodes` 目录中运行：

```powershell
git clone https://github.com/Saganaki22/Zonos2_TTS-ComfyUI.git
..\venv\Scripts\python.exe Zonos2_TTS-ComfyUI\install.py
```

Linux 便携版或 venv 环境可运行：

```bash
git clone https://github.com/Saganaki22/Zonos2_TTS-ComfyUI.git
../venv/bin/python Zonos2_TTS-ComfyUI/install.py
```

安装或更新后请重启 ComfyUI。`install.py` 只安装缺失的轻量依赖，不会替换 ComfyUI 已有的 `torch`、`torchaudio` 或 `transformers`。

</details>

<details>
<summary><strong>模型与自动下载</strong></summary>

模型统一存放在：

```text
ComfyUI/models/zonos2/
├── zonos2-bf16.safetensors
├── dac_44khz/
│   ├── .gitattributes
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   └── README.md
└── speaker_encoder/
    ├── config.json
    ├── configuration_ecapa_tdnn.py
    ├── feature_extraction_ecapa_tdnn.py
    ├── model.safetensors
    ├── modeling_ecapa_tdnn.py
    ├── preprocessor_config.json
    ├── tokenizer_config.json
    └── tokenizer_ecapa_tdnn.py
```

预设加载器会从 [drbaph/ZONOS2-BF16](https://huggingface.co/drbaph/ZONOS2-BF16) 下载全部三个资产：

- 选择预设时下载 `zonos2-bf16.safetensors`。
- 首次加载模型包时下载 `dac_44khz/`。
- 首次使用声音克隆时按需下载 `speaker_encoder/`。

启用 `download_if_missing` 后允许自动下载。若要完全离线运行，请先将所有必要资产放入上述路径，然后关闭此选项。

节点自带官方架构配置 `assets/params.json`，加载时始终使用它验证检查点中的名称和形状。手动加入的 `.safetensors` 文件也会出现在模型下拉列表中。

请将完整的 `dac_44khz/` 和 `speaker_encoder/` 目录上传到模型仓库，不要按照简化目录树手动重建。运行时，DAC 直接需要 `config.json` 和 `model.safetensors`。说话人编码器通过 Transformers 远程代码机制加载，因此直接需要 `config.json`、`model.safetensors`、`configuration_ecapa_tdnn.py` 和 `modeling_ecapa_tdnn.py`。预处理器、特征提取器、分词器、README 和 Git 元数据文件不由当前节点推理路径直接调用，但保留完整上游目录能够维持一个有效且可复用的 Hugging Face 模型包。

</details>

<details>
<summary><strong>节点与设置</strong></summary>

### ZONOS2 Model Loader

| 输入 | 默认值 | 可选值 | 说明 |
|---|---|---|---|
| `model` | ZONOS2 BF16 预设 | 预设与本地 safetensors | 从 `ComfyUI/models/zonos2` 加载。 |
| `dtype` | `auto` | `auto`、`bf16`、`fp16` | `auto` 保留检查点数据类型；显式选择会在加载时转换浮点张量。 |
| `attention` | `auto` | `auto`、`SDPA`、`flash_attention` | `auto` 在兼容时使用 FlashAttention，否则使用 SDPA。 |
| `download_if_missing` | `true` | 布尔值 | 下载缺失的检查点、DAC 和说话人编码器。 |

模型、数据类型或注意力后端发生变化时，节点会先硬卸载旧模型，再加载新模型。

### ZONOS2 Voice Generation

接收 `zonos2_model` 和 UTF-8 文本，返回 44.1 kHz 单声道 ComfyUI `AUDIO`。

### ZONOS2 Voice Clone

接收 `zonos2_model`、UTF-8 文本和必需的 ComfyUI `reference_audio` 连线。ZONOS2 会直接从音频中提取官方的 2048 维 ECAPA-TDNN 说话人嵌入，因此不需要参考文本。

| 克隆设置 | 默认值 | 说明 |
|---|---:|---|
| `reference_audio` | 必需 | 推荐使用 5–30 秒、单人且干净的语音。节点最多接受 60 秒，超出部分会被裁剪，并在 CLI 中显示警告。 |
| `clean_speaker_background` | `true` | 干净语音请启用；存在明显房间噪声或环境声时请关闭。 |
| `accurate_mode` | `true` | 更侧重说话人相似度；关闭后条件约束更宽松，可能更有表现力。 |

### 采样控制

| 控制项 | 默认值 | 范围 | 说明 |
|---|---:|---:|---|
| `max_new_tokens` | 1024 | 32–6000，步长 8 | 最大 DAC 代码帧数；数值越高，可生成的语音越长，但耗时和 KV 缓存占用也越高。 |
| `temperature` | 1.15 | 0–2 | 采样随机度；`0` 为贪心采样。 |
| `top_k` | 106 | 0–1026 | 仅保留概率最高的 K 个 token；`0` 表示关闭。 |
| `top_p` | 0.0 | 0–1 | 核采样阈值；`0` 表示关闭。 |
| `min_p` | 0.18 | 0–1 | 移除概率低于最高概率一定比例的 token；`0` 表示关闭。 |
| `repetition_window` | 50 | 0–512 | 检查重复的最近帧数；`0` 表示关闭。 |
| `repetition_penalty` | 1.2 | 1–2 | 抑制重复音频 token；`1` 表示关闭惩罚。 |
| `repetition_codebooks` | 8 | -1–9 | 应用重复惩罚的码本数量；`-1` 表示全部，`0` 表示不应用。 |
| `seed` | 0 | 0–9223372036854775807 | 正数可让相同输入重复生成一致结果；`0` 使用当前随机状态。 |

### ZONOS2 条件控制

每个条件下拉框都包含 `default`，表示不对该特征进行条件约束。

| 控制项 | 已发布桶 | 默认值 |
|---|---|---|
| `speaking_rate` | 8 个 UTF-8 字节/秒范围，从 `0–8` 到 `40+` | `default` |
| `loudness_lufs` | 12 个范围，从低于 `-50` 到 `-5+` LUFS | `default` |
| `estimated_snr` | 12 个范围，从低于 `0` 到 `60+` dB | `default` |
| `maximum_pause` | 12 个范围，从 `0–0.5` 到 `5.5–6` 秒 | `default` |
| `estimated_bandlimit_hz` | 8 个范围，从 `495.3–3433` 到 `21062–24000` Hz | `default` |
| `leading_silence` | 8 个范围，从 `0–0.05` 到 `4+` 秒 | `default` |
| `trailing_silence` | 8 个范围，从 `0–0.05` 到 `4+` 秒 | `0.25–0.5` |

</details>

<details>
<summary><strong>参考音频指南</strong></summary>

- 推荐长度：5–30 秒干净、连续、单人语音。
- 本节点包接受的最大长度：60 秒。
- 更长的输入会在重采样和提取嵌入前裁剪为最前面的 60 秒。
- ComfyUI CLI 会显示原始时长和具体裁剪操作。
- 少于 5 秒的参考音频也会显示建议警告。
- 请避免音乐、多人重叠、严重降噪伪影、强混响和长时间静音。

60 秒上限是本集成针对内存和延迟设置的实用保护措施，并非上游 ZONOS2 模型公开说明的架构硬限制。

</details>

<details>
<summary><strong>Transformers 兼容性</strong></summary>

本节点包已使用 **Transformers 5.3.0** 完成端到端测试。DAC 解码与说话人编码器前向传播还通过了以下版本：

`5.0.0`、`5.2.0`、`5.3.0`、`5.5.4`、`5.12.0`

支持并经过测试的范围：**`transformers>=5.0.0,<=5.12.0`**。

本节点包不支持 Transformers 4.x，因为常见的当前 ComfyUI 环境与这些旧版本所需的 NumPy 和 `huggingface-hub` 约束存在冲突。`install.py` 会报告当前版本是否位于测试范围内，但绝不会自动替换 Transformers。

</details>

<details>
<summary><strong>显存管理与进度</strong></summary>

ZONOS2 主模型、DAC 解码器和按需加载的说话人编码器都会作为真实 PyTorch 模块注册到 ComfyUI/AIMDO。

- ComfyUI 可以将模型包卸载到 CPU，并在下次运行时恢复。
- AIMDO 显示条反映真实 GPU 张量驻留状态。
- 使用完全相同的加载设置时会恢复现有模型包。
- 更改模型、数据类型或注意力后端时，旧模型会取消注册，张量会移至 `meta`，随后清理引用、执行垃圾回收并清空加速器缓存。
- 加载和生成均使用原生 ComfyUI 进度条及 CLI `tqdm`。

实测主 BF16 模型和 DAC 加载后的 CUDA 分配约为 14.7 GiB。还需为 KV 缓存、说话人编码器、ComfyUI 和其他节点预留额外显存。

</details>

<details>
<summary><strong>故障排除</strong></summary>

**模型下拉菜单下载失败或返回 404**

确认 [drbaph/ZONOS2-BF16](https://huggingface.co/drbaph/ZONOS2-BF16) 中已经包含 `zonos2-bf16.safetensors`、`dac_44khz/` 和 `speaker_encoder/`。上传尚未完成时，部分文件可能暂时无法下载。

**FlashAttention 不可用**

选择 `attention: auto` 或 `SDPA`。FlashAttention 未安装、设备不是 CUDA 或数据类型不兼容时，`auto` 会回退到 SDPA。

**CUDA 显存不足**

卸载其他大型 ComfyUI 模型、降低 `max_new_tokens`、使用 ComfyUI 卸载功能，或在分配失败后重启 ComfyUI。ZONOS2 BF16 与 DAC 在生成缓存增长前约占 14.7 GiB。

**声音克隆相似度较低**

使用 5–30 秒干净的单人语音，让 `clean_speaker_background` 与录音环境一致，并保持 `accurate_mode` 启用以获得更高相似度。

**参考音频超过 60 秒**

节点会保留前 60 秒并打印类似警告：`Reference audio is 75.00 seconds; the node accepts at most 60.0 seconds. Clipping to the first 60.0 seconds.`

**输出过早结束**

提高 `max_new_tokens`。模型发出音频结束 token 后仍可能提前停止。

**Transformers 导入或模型加载失败**

检查 `install.py` 的启动输出，并使用测试范围 `5.0.0–5.12.0` 内的版本。推荐基准版本为 `5.3.0`。

**改变模型、数据类型或注意力后端后全部重新加载**

这是预期行为。旧张量会被硬卸载，以免 AIMDO 注册残留或显存未释放。

</details>

<details>
<summary><strong>示例工作流</strong></summary>

在 ComfyUI 中加载 `example_workflows/zonos2_tts_example_workflow.json`。其中包含普通生成与声音克隆两个分支。运行克隆分支前，请在 `LoadAudio` 节点中选择自己的音频文件。

</details>

<details>
<summary><strong>许可证与负责任使用</strong></summary>

- 原始 ZONOS2 模型权重采用 [Apache License 2.0](https://huggingface.co/Zyphra/ZONOS2)。
- 本 ComfyUI 集成代码采用 MIT License。
- DAC、说话人编码器和其他依赖仍受各自上游许可证约束。

Apache-2.0 模型许可证与本项目的可接受使用政策彼此独立。无论许可证允许哪些用途，都不得将本项目用于恶意冒充、欺诈、误导、骚扰、未经同意的声音克隆或造成伤害。

</details>

<details>
<summary><strong>引用</strong></summary>

如果本模型对您的学术研究有帮助，请引用：

```bibtex
@misc{zyphra2025zonos,
  title     = {Zonos V2 Technical Report},
  author    = {Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge},
  year      = {2026},
}
```

</details>

## 致谢

- [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2)
- [Zyphra/ZONOS2 官方模型](https://huggingface.co/Zyphra/ZONOS2)
- [ComfyUI 原生 BF16 模型包](https://huggingface.co/drbaph/ZONOS2-BF16)
- [Descript DAC 44.1 kHz](https://huggingface.co/descript/dac_44khz)
