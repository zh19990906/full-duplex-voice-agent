# 全双工语音助手 V1 落地设计

## 1. 文档目的

本文定义 `full-duplex-voice-agent` 从“真实模型可运行的语音问答原型”演进为“单机、单用户、效果优先的全双工语音助手演示版”的落地方案。

目标体验对齐 GPT Live 的交互方向，但不追求相同的基础模型能力和极限时延。V1 优先验证实时编排能力：持续监听、用户附和、语义打断、暂停恢复、方向修订和同声传译。

## 2. V1 范围

### 2.1 目标能力

V1 必须支持：

1. 助手播放语音时持续监听用户麦克风。
2. 用户说“嗯、对、继续”等附和时，不取消当前回答。
3. 用户说“等一下”时立即暂停，并保留尚未播放的内容。
4. 用户说“继续”或语义等价表达时，从最近稳定短语恢复。
5. 用户说“不对，我说的是北京，不是上海”时，取消旧方向并按修订意图重新生成。
6. 用户提出新问题时，停止当前回答并进入新请求。
7. 普通问答包含单次翻译，不为单次翻译建立持久任务模式。
8. 用户明确提出持续翻译需求时，进入同声传译模式；语言方向由 LLM 判断并可动态切换。
9. 通过真实硬件时间线验证时延、取消可靠性和陈旧音频泄漏。

### 2.2 交付形态

- 单机部署。
- 单个活跃用户会话。
- 一张 NVIDIA RTX PRO 5000 72GB Blackwell GPU。
- 浏览器前端，端口为 `8001`。
- V1 验收要求用户佩戴耳机或耳麦。
- 当前进程生命周期内支持暂停和恢复；服务重启后恢复、跨设备恢复不属于 V1。

### 2.3 非目标

- 多租户并发和横向扩展。
- 开放式扬声器环境下的完整声学回声消除。
- 训练新的端到端语音模型。
- 大规模微服务、消息队列或多节点 GPU 调度。
- 助手在用户长时间说话时主动播放“嗯、我在听”等语音附和。协议保留该能力，但实现放到 V1 之后。

## 3. 当前能力基线

当前仓库已经具备较好的模块化基础，但必须区分真实能力、离线验证和模拟验证。

### 3.1 已真实验证

- Qwen2.5-14B-Instruct 可在本地 GPU 流式输出 token。
- X2-Turn-4B 可对完整音频文件输出转写和帧级 turn 标签。
- Fun-CosyVoice3-0.5B 可在独立 Python 环境中生成 24kHz 单声道 PCM16 音频。
- X2-Turn、Qwen、CosyVoice 的离线串行链路可生成完整回复 WAV。
- 浏览器文字输入可经过真实 Qwen 和 CosyVoice 播放语音。
- WebSocket、浏览器 PCM 播放和流式文本展示已经建立基本通路。

### 3.2 仅由模拟组件验证

- 用户附和不取消生成。
- 打断时取消生成和 TTS。
- 陈旧音频帧为零。
- 任务暂停和恢复。
- 流式翻译工作流。

### 3.3 尚未形成真实在线链路

- 浏览器连续 PCM 麦克风输入。
- faster-whisper 增量转写。
- X2-Turn rolling inference。
- 声学事件与语义动作融合。
- 可抢占的 Qwen、TTS 和浏览器播放器。
- 基于真实播放游标的暂停和恢复。
- 增量文本到流式 TTS。
- 真实同声传译。

因此，V1 的核心任务不是增加更多抽象，而是让真实模型走过唯一的实时运行时。

## 4. 架构决策

采用“事件驱动的单机实时运行时”。模型保持可替换，控制逻辑集中在实时控制平面；仅在依赖隔离或阻塞推理需要时使用模型 worker，不进行微服务化。

不继续把业务逻辑堆入 `scripts/run_real_server.py`。该脚本只负责配置加载、应用启动、WebSocket/HTTP 传输和静态页面托管。

目标架构：

```text
Browser AudioWorklet
  │  PCM16 / 16kHz / mono / 20ms frame
  ▼
TransportGateway
  │
  ▼
AudioIngress
  ├── jitter buffer / sequence / timestamp
  ├── Streaming ASR worker
  └── X2-Turn rolling worker
          │
          ▼
   SpeechEventFusion
     ├── 可逆快速动作：duck / tentative pause
     └── SemanticPolicyEngine
                  │
                  ▼
        ConversationController
          ├── CHAT workflow
          └── INTERPRETATION workflow
                  │
         ┌────────┴────────┐
         ▼                 ▼
     Qwen stream      Stable translation
         │                 │
         └──────┬──────────┘
                ▼
          TextSegmenter
                ▼
        CosyVoice worker
                ▼
      PlaybackCoordinator
                ▼
         Browser AudioWorklet
```

## 5. 核心组件

### 5.1 `RealtimeSessionRuntime`

每个活跃会话拥有一个实时运行时，负责组合各组件并维护生命周期。它是唯一真实业务入口，不直接实现模型细节。

职责：

- 启动和停止会话 worker。
- 接收音频帧和控制事件。
- 发布内部领域事件。
- 将控制器动作提交给执行器。
- 维护当前 `response_id` 和 `generation_epoch`。
- 提供会话快照、健康信息和时间线。

### 5.2 `TransportGateway`

只负责传输：

- 创建、关闭和重连会话。
- 接收浏览器二进制 PCM 帧。
- 接收文字消息和控制消息。
- 发送转写、token、音频、状态和错误事件。
- 不判断附和、打断、恢复或翻译模式。

### 5.3 `AudioIngress`

- 校验帧头、序号、采样率和时间戳。
- 维护短时 jitter buffer。
- 检测丢帧和过期帧。
- 将同一份音频并行投递到快速音频活动检测、ASR 和 X2-Turn。
- 为所有下游结果保留原始 capture timestamp。

### 5.4 `SpeechEventFusion`

融合以下信号：

- 音频能量或轻量 VAD 的 speech-start。
- faster-whisper partial/final transcript。
- X2-Turn 的 speaking、idle、turn-end 和附和候选。
- 当前话轮持有者、助手是否正在提问、当前播放状态。

它只生成候选领域事件，不直接改变业务状态。

### 5.5 `SemanticPolicyEngine`

使用独立的 1.5B～3B Qwen Instruct 模型，输出严格 JSON，不生成用户可见内容。

输入包括：

- 当前模式、话轮和响应状态。
- 助手最近一句、当前行为类型和未播放文本摘要。
- 用户增量或最终转写。
- X2-Turn 候选标签和置信度。
- 当前任务检查点。

输出动作：

```text
BACKCHANNEL
ANSWER
PAUSE
RESUME
REVISE
NEW_REQUEST
MODE_SWITCH
UNCERTAIN
```

约束：

- 最大输出约 64 token。
- 只允许 schema 中的枚举和字段。
- 目标响应时间不超过 300ms。
- 超时、解析失败或低置信度统一转为 `UNCERTAIN`。
- `UNCERTAIN` 的安全行为是暂停并澄清，不是继续旧内容。

### 5.6 `ConversationController`

控制器是唯一控制平面。模型发出信号，控制器决定动作，执行器负责副作用。

不继续使用单一枚举表达所有状态，改成三个正交维度：

```text
mode:      CHAT | INTERPRETATION
floor:     NONE | USER | ASSISTANT | OVERLAP
response:  IDLE | GENERATING | PLAYING | DUCKED | PAUSED | CANCELLING
```

### 5.7 `ResponsePipeline`

- 流式消费 Qwen token。
- 使用语言感知分段器提交稳定短句。
- 维护有界 TTS 队列并实施背压。
- 将所有 token、片段和音频绑定到 response/epoch。
- 支持取消和迟到结果过滤。

### 5.8 `PlaybackCoordinator`

- 向浏览器发送音频片段和控制指令。
- 维护已合成、已发送和已确认播放的游标。
- 支持 `DUCK`、`RESTORE`、`PAUSE_RESPONSE` 和 `STOP_RESPONSE`。
- 根据浏览器 ACK 更新真实播放位置。
- 拒绝旧 epoch 音频。

### 5.9 `SessionStateStore`

V1 使用进程内存保存：

- 当前模式和语言方向。
- 当前和最近暂停的 response。
- 已生成文本、稳定片段、未播放文本。
- 已合成片段和播放游标。
- 对话历史和结构化任务检查点。

服务重启持久化明确延期，不与 V1 混合实现。

## 6. 附和与重叠说话

附和是一条一级快速路径，不是关键词特殊分支。

```text
助手正在播放 + 用户开口
        │
        ├── 疑似附和
        │      ├── 立即轻微 duck
        │      ├── 保持 LLM 生成
        │      ├── 保持 TTS 队列
        │      └── 等待 ASR + 策略确认
        │
        └── 疑似打断
               ├── 立即 tentative pause
               └── 等待语义动作确认
```

语义确认规则：

- 助手陈述时用户说“嗯、对、继续”：`BACKCHANNEL`。
- 助手刚提出确认问题时用户说“对”：`ANSWER`。
- “对，不过我说的是北京”：整体判为 `REVISE`。
- “嗯……等一下”：最终判为 `PAUSE`。

`BACKCHANNEL` 不创建正式用户消息，默认只写入时间线和会话元数据；当肯定或否定构成助手问题的答案时，才写入正式对话历史。

V1 之后可实现 `ASSISTANT_BACKCHANNEL`，让助手在用户长叙述时主动附和。V1 只冻结事件和状态，不播放主动附和音频。

## 7. 暂停、打断和恢复

### 7.1 四层游标

每个回答维护：

```text
generated_cursor    LLM 已生成位置
committed_cursor    稳定文本片段位置
synthesized_cursor  已合成片段位置
played_cursor       浏览器实际播放位置
```

每个稳定片段至少包含：

```json
{
  "response_id": "response-12",
  "generation_epoch": 7,
  "segment_id": 4,
  "text": "第二个值得去的地方是故宫。",
  "audio_status": "PLAYING",
  "sample_offset": 18240
}
```

### 7.2 `PAUSE`

1. 浏览器立即停止当前播放推进。
2. 服务端停止提交新的 TTS 片段。
3. LLM 只允许运行到下一个稳定短句边界；未播文本使用有界缓冲。
4. 保存 response、segment 和 played cursor。
5. 不把暂停当作新的用户问题。

### 7.3 `RESUME`

1. 策略模型选择最近或语义指定的暂停 response。
2. 当前片段若只播放一部分，从该稳定短语开头重播。
3. 有缓存音频时直接播放；只有文本时重新合成。
4. 缓冲文本耗尽后，使用原始问题、已完成文本和最后稳定片段继续生成。

验收允许最多重复或丢失一个短语，不允许整段重说，也不从音素中间恢复。

### 7.4 `REVISE` 与 `NEW_REQUEST`

- `REVISE`：取消当前生成，清空未播队列，归档旧回答，将修订后的用户意图写入上下文并重新生成。
- `NEW_REQUEST`：取消当前回答并进入新问题；旧回答保留为可恢复历史，但不会自动恢复。

## 8. 防止陈旧输出

所有生成和输出携带：

```text
session_id
response_id
generation_epoch
segment_id
```

每次 `REVISE`、`NEW_REQUEST` 或不可恢复取消都增加 `generation_epoch`。以下边界都必须校验 epoch：

- LLM token 发布。
- 文本片段入 TTS 队列。
- CosyVoice worker 音频返回。
- WebSocket 音频发送。
- 浏览器音频入播放队列。

底层推理即使不能立即停止，迟到结果也不能越过这些边界。V1 的 stale audio 验收值为零。

## 9. 浏览器音频设计

### 9.1 输入

使用 AudioWorklet 代替 MediaRecorder：

- 16kHz。
- 单声道。
- PCM16。
- 每帧 20ms。
- 帧头包含 sequence 和 capture timestamp。
- `getUserMedia` 开启浏览器 echo cancellation、noise suppression 和 auto gain control。

耳麦是 V1 验收前提，因此不额外承担开放扬声器 AEC 的产品保证。

### 9.2 输出

浏览器播放器需要：

- 使用 GainNode 在 50～100ms 内降低音量。
- 记录每个音频节点对应的 response/epoch/segment。
- 支持恢复、暂停和按 response 停止。
- 周期性回传播放 ACK 和 sample offset。
- 对旧 epoch 音频拒绝排队。

## 10. 真实流式 ASR 与 X2-Turn

### 10.1 faster-whisper rolling ASR

faster-whisper-large-v3 使用滚动窗口适配：

1. 缓存最近数秒音频。
2. 每 200～400ms 增量解码。
3. 比较连续假设，提取稳定公共前缀。
4. 只发布新增稳定文本。
5. X2-Turn 给出 turn-end 后执行最终解码。
6. partial 携带 revision id，前端替换不稳定尾部。

### 10.2 X2-Turn rolling inference

- 使用 1～3 秒滑动上下文。
- 约 100～200ms 更新一次，不对每个 20ms 帧执行完整推理。
- 使用滞回窗口抑制 idle/speaking 抖动。
- 连续帧确认 turn-end。
- 声学附和只产生候选事件。

X2 rolling wrapper 是 V1 的首要技术风险，必须先做真实实时 benchmark。如果不能满足更新频率：

- speech-start 使用轻量活动检测。
- turn-end 使用 ASR 静音和文本稳定性。
- X2-Turn 降级为低频辅助信号。
- 领域事件和控制器接口保持不变。

X2-Turn 不作为 ASR、VAD、话轮和语义意图的唯一来源。

## 11. 对话生成与流式 TTS

目标链路：

```text
Qwen token stream
→ language-aware TextSegmenter
→ bounded segment queue
→ CosyVoice worker
→ audio chunk queue
→ browser
```

中文分段策略：

- 优先使用逗号、句号、问号和语义短语边界。
- 首段约 12～24 个汉字即可提交。
- 后续片段适当加长以改善韵律。
- 避免极短片段频繁调用 TTS。

英文按标点、从句和词数分段。TTS 队列有上限，TTS 变慢时对 LLM 提交施加背压。

CosyVoice 继续运行在独立环境和进程中。worker 协议增加 request id、cancel 和完成状态。即使底层推理不能立刻取消，主进程仍按 epoch 丢弃输出。

## 12. 同声传译

### 12.1 模式

持久模式只有：

```text
CHAT
INTERPRETATION
```

单次翻译属于普通 CHAT 请求，不创建独立任务模式。只有策略模型识别出“接下来、持续、实时、一直翻译”等持续性语义时才进入 INTERPRETATION。

模糊表达默认按单次请求处理，不自动进入持续同传。

### 12.2 翻译模型

不增加专用翻译模型。Qwen2.5-14B-Instruct 同时承担：

- 普通问答。
- 单次翻译。
- 同传模式下的增量翻译。
- 源语言检测和目标语言遵循。

策略小 LLM 只负责模式和动作，不生成最终译文。

### 12.3 稳定提交

```text
ASR partial
→ StablePrefixCommitter
→ SourceSegment
→ Qwen strict translation prompt
→ TranslationSegment
→ TextSegmenter
→ CosyVoice
```

规则：

- 只翻译连续两次 ASR 假设中一致的稳定前缀。
- 已经开始播放的译文不可回滚。
- 不稳定尾部等待更多语音。
- source segment 与 translation segment 一一关联，防止重复。
- 切换目标语言从下一个稳定源片段生效。
- 原话纠正只修改尚未播放的译文；已播内容通过简短纠正补偿。

V1 允许增加约 0.5～1 秒稳定等待，以换取更低的错误率。

## 13. 模型与资源调度

V1 模型组合：

| 职责 | 模型 |
|---|---|
| 增量 ASR | faster-whisper-large-v3 |
| 话轮与附和候选 | X2-Turn-4B |
| 语义动作策略 | 1.5B～3B Qwen Instruct |
| 对话与翻译 | Qwen2.5-14B-Instruct |
| 流式语音合成 | Fun-CosyVoice3-0.5B |

实时优先级：

```text
1. 音频停止和 duck
2. speech-start 与 X2-Turn
3. 策略小 LLM
4. 首段 TTS
5. 主 LLM 生成
6. 后续 TTS 和后台状态写入
```

当前 `RealtimeScheduler` 只能决定等待队列的启动顺序，不能抢占正在运行的任务。V1 必须增加取消令牌、deadline 和 active-task preemption，或者由每个工作流显式管理可取消 asyncio task。

启动阶段记录各模型加载前后显存。预留 KV cache、CUDA context 和临时 tensor 空间；显存超过安全预算时拒绝继续加载并给出明确诊断。

策略模型是否放 GPU 由真实 benchmark 决定。如果与主 Qwen 争抢导致超过 300ms，可将量化策略模型迁到 CPU，或后续引入支持请求取消和优先级的推理服务。

## 14. 事件协议

内部领域事件和浏览器传输事件分层，不共享任意字典。

统一内部信封：

```json
{
  "protocol_version": 1,
  "event": "BACKCHANNEL_CONFIRMED",
  "event_id": "evt-123",
  "session_id": "session-1",
  "sequence": 418,
  "capture_timestamp": 123.41,
  "server_timestamp": 123.48,
  "response_id": "response-7",
  "generation_epoch": 3,
  "payload": {}
}
```

要求：

- schema 可验证。
- sequence 可检测乱序和丢失。
- 双时间戳可拆分网络、模型和播放延迟。
- protocol version 支持兼容演进。
- 音频二进制帧具有固定 metadata 头部。
- 核心链路不得发布未定义结构的字典。

## 15. 容错与降级

| 故障 | V1 行为 |
|---|---|
| X2-Turn 超时 | 使用活动检测和 ASR 静音/稳定文本判断话轮 |
| ASR 失败 | 停止本轮语义决策并提示用户重说 |
| 策略 LLM 超时 | 安全暂停并进入 UNCERTAIN |
| 主 LLM 失败 | 保留会话状态并返回明确错误 |
| TTS 失败 | 保留文字回复，前端显示语音失败 |
| CosyVoice worker 退出 | 自动重启一次并重新提交未播放文本 |
| WebSocket 短暂断线 | 保留进程内会话，重连后发送 snapshot |
| 浏览器播放失败 | 停止继续推送，防止队列无限增长 |

取消、停止和关闭操作必须幂等。

## 16. 性能与验收指标

| 指标 | V1 目标 |
|---|---:|
| 用户开始插话到助手 duck | ≤ 100ms |
| 确认打断到旧音频完全停止 | ≤ 250ms |
| 附和结束到恢复正常音量 | ≤ 300ms |
| 用户话轮结束到首个 LLM token | ≤ 800ms |
| 用户话轮结束到首段可播放音频 | ≤ 1.5s |
| 同传稳定源片段到首段译音 | ≤ 2s |
| stale audio chunk | 0 |
| 恢复位置误差 | 最多一个短语 |

这些指标必须由真实浏览器、耳麦和本地模型时间线生成，不接受手工填写时间戳。

## 17. 测试体系

### 17.1 分层

1. 单元测试：状态、动作、游标、epoch 和 schema。
2. 确定性集成测试：模拟模型，验证事件顺序和取消。
3. 录音实时回放：按原始时间节奏注入真实音频。
4. 模型集成：真实模型但不依赖物理麦克风。
5. 硬件 E2E：浏览器、耳麦、播放和真实打断。
6. 稳定性：连续运行 30～60 分钟并记录显存和队列。

### 17.2 必测场景

- 陈述期间“嗯嗯”不取消回答。
- 助手提问后的“对”作为有效回答。
- “等一下”在 250ms 内停止，随后“继续”从短语边界恢复。
- “不是上海，是北京”切换方向且旧音频泄漏为零。
- 新问题取消当前回答，旧回答只进入历史。
- 单次翻译保持 CHAT。
- 持续翻译进入 INTERPRETATION。
- 同传切换目标语言从下一个稳定片段生效。
- X2、策略模型、TTS 和 WebSocket 故障按定义降级。

测试报告必须标注 `unit`、`simulated`、`model-integration` 或 `hardware-e2e`，避免以测试数量代替真实能力完成度。

## 18. 实施阶段

### 阶段 0：协议和基线冻结，2～3 人日

- 冻结事件信封、动作 schema 和各类 ID。
- 记录真实模型显存、RTF 和首结果延迟。
- 建立统一 `RealtimeSessionRuntime` 组合入口。
- 输出真实能力矩阵。

退出条件：协议测试通过，基准可重复。

### 阶段 1：双向 PCM 和播放器，3～5 人日

- AudioWorklet 输入。
- WebSocket 二进制帧。
- GainNode duck。
- 可控播放队列和 ACK。

退出条件：边播边录稳定，100ms 内完成 duck。

### 阶段 2：流式 ASR 和 X2，5～10 人日

- faster-whisper rolling wrapper。
- X2-Turn rolling wrapper。
- 多信号融合。
- 真实时间回放 benchmark。

退出条件：产生 partial/final、speech-start、turn-end 和 backchannel candidate，RTF 小于 1。

### 阶段 3：策略 LLM 和控制器，4～7 人日

- 结构化策略输出。
- 正交状态模型。
- 超时和低置信度降级。
- 完整动作测试集。

退出条件：关键动作分类达到基准要求，异常时安全暂停。

### 阶段 4：增量 Qwen 到 CosyVoice，5～8 人日

- 文本稳定分段。
- 有界 TTS 队列和背压。
- worker 取消协议。
- epoch 全链路过滤。

退出条件：首音频 ≤ 1.5s，陈旧音频为零。

### 阶段 5：附和、暂停和恢复，5～8 人日

- 四层游标。
- 附和快速路径。
- 短语边界恢复。
- 修订和历史 response 选择。

退出条件：全部附和、打断和恢复场景通过。

### 阶段 6：同声传译，5～8 人日

- CHAT/INTERPRETATION 模式。
- StablePrefixCommitter。
- Qwen 严格翻译工作流。
- 语言方向切换。

退出条件：稳定源片段到译音 ≤ 2s，无重复片段。

### 阶段 7：稳定性和演示封装，3～5 人日

- worker watchdog。
- 显存和队列监控。
- 重连 snapshot。
- 30～60 分钟稳定测试。
- 一键启动和真实 benchmark 报告。

单人总体约 5～8 周。多人并行时必须先冻结协议，浏览器音频、模型 wrapper 和控制器才可独立开发。

## 19. 针对当前代码的建设性建议

1. 保留 Adapter、Controller、EventBus、TaskState 和 benchmark 的模块化方向，不推倒重写。
2. 停止维护“设计架构”和 `run_real_server.py → RealModelSession` 两条真实路径，将真实模型迁入唯一运行时。
3. 在迁移过程中逐步合并 `core/interfaces`、`adapters/base`、`backend` 和重复 pipeline，不进行脱离功能目标的大重构。
4. 内部领域事件、API 事件和浏览器事件分层并版本化。
5. 配置文件改为真实 Qwen、X2、faster-whisper 和 CosyVoice worker；移除业务代码中的固定 `/home` 路径。
6. 主环境和 CosyVoice 环境分别锁定依赖，启动前验证 CUDA、模型路径、采样率和 worker 可执行性。
7. 取消成为所有长任务的一级接口，定义 cooperative cancel、强制丢弃和 worker 重启三层语义。
8. 真实硬件 benchmark 成为发布门禁，模拟测试不冒充产品能力。
9. 单机阶段只使用 asyncio 控制平面和必要 worker，不引入 Redis、消息队列和多节点调度。
10. 优先优化首反馈、首音频和取消可靠性，而不是总 token 吞吐。
11. 浏览器播放游标、ACK、duck 和停止是实时系统的一部分，不能视为纯 UI 细节。
12. 为事件协议和关键架构变更建立 ADR，避免后续并行开发破坏控制边界。

## 20. 成功定义

V1 完成不是“所有模型能独立运行”，也不是“模拟测试全部通过”，而是：

- 一个用户佩戴耳麦，在浏览器中与本地模型持续双向语音交互。
- 用户可以自然附和而不误打断。
- 暂停、继续、纠正和新请求由语义决定。
- 旧生成和旧音频不会在新回答中泄漏。
- 主 LLM 可自动处理单次翻译，并在持续指令下进入同声传译。
- 所有关键行为具有真实时间线、可重复 benchmark 和明确降级路径。
