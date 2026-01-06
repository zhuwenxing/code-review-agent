# Code Review Agent

一个强大的智能代码审查工具，支持 Claude 和 Gemini 双 LLM 引擎，具有增量审查、断点续传、并行执行等特性。

## 主要特性

- **多 LLM 支持** - 支持 Claude Agent SDK 和 Gemini，可灵活切换
- **增量审查** - 基于文件内容哈希，仅审查变更的文件，大幅提升效率
- **断点续传** - 自动保存审查状态，中断后可从断点继续
- **代码库探索** - 自动分析代码库模式，生成项目特定的审查规则
- **智能 .gitignore 支持** - 完整支持 .gitignore 语法，包括嵌套 .gitignore 文件
- **并行审查** - 多文件并发审查，可配置并发数
- **大文件分块处理** - 自动将大文件分割为代码块进行审查，然后合并结果
- **层级目录输出** - 审查报告保持与源代码相同的目录结构
- **中文支持** - 审查报告默认使用中文输出

## 安装

```bash
# 克隆仓库
git clone https://github.com/zhuwenxing/code-review-agent.git
cd code-review-agent

# 创建虚拟环境并安装依赖
uv venv -p 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# 安装为 CLI 工具
uv pip install -e .
```

## 使用方法

### 基本用法

```bash
# 审查当前目录（使用 Gemini，默认）
code-review-agent

# 审查指定目录
code-review-agent /path/to/your/code

# 使用 Claude 进行审查
code-review-agent --agent claude
```

### 增量审查

默认启用增量审查模式，仅审查自上次运行以来变更的文件：

```bash
# 正常运行（自动增量审查当前目录）
code-review-agent

# 强制全量审查（忽略之前的状态，可与路径参数一同使用）
code-review-agent --force-full

# 禁用断点续传（可与路径参数一同使用）
code-review-agent --no-resume
```

### 高级选项

```bash
code-review-agent /path/to/code \
  --agent claude \
  --extensions py,go,js,ts \
  --output-dir reviews \
  --concurrency 10 \
  --max-files 100 \
  --skip-explore \
  --debug
```

### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | - | 当前目录 | 要审查的目录路径 |
| `--agent` | `-a` | `gemini` | LLM 引擎：`claude` 或 `gemini` |
| `--extensions` | `-e` | `py,go,js,ts,java,cpp,c,h` | 要审查的文件扩展名 |
| `--output-dir` | `-o` | `reviews` | 审查报告输出目录 |
| `--max-files` | `-m` | 无限制 | 最大审查文件数量 |
| `--concurrency` | `-c` | `5` | 并发审查工作线程数 |
| `--retry` | `-r` | `2` | 失败重试次数 |
| `--chunk-lines` | - | `500` | 大文件分块的行数 |
| `--large-file-threshold` | - | `800` | 触发分块审查的行数阈值 |
| `--skip-explore` | - | `False` | 跳过代码库探索阶段 |
| `--force-full` | - | `False` | 强制全量审查，忽略之前的状态 |
| `--resume/--no-resume` | - | `True` | 启用/禁用断点续传 |
| `--debug` | - | `False` | 启用调试模式，显示完整堆栈跟踪 |

## 输出结构

审查报告保持与源代码相同的目录结构：

```
reviews/
├── SUMMARY.md                    # 汇总报告（含统计数据）
├── .review_state.json           # 审查状态文件（用于增量/续传）
├── src/
│   ├── main.py.md               # src/main.py 的审查报告
│   └── utils/
│       └── helper.py.md         # src/utils/helper.py 的审查报告
└── ...
```

## 审查流程

工具分 4 个阶段运行：

1. **阶段 1：探索** - 分析代码库以了解模式并生成特定的审查规则
2. **阶段 2：发现** - 查找所有匹配指定扩展名的文件（应用 .gitignore 过滤，检测变更文件）
3. **阶段 3：审查** - 并行审查文件，每个文件完成后立即保存
4. **阶段 4：汇总** - 生成最终汇总报告

## 标准审查领域

工具检查以下方面：

1. **安全性** - 漏洞、注入风险
2. **缺陷** - 潜在 bug、边界情况、nil/null 问题
3. **并发** - 竞态条件、死锁、goroutine 泄漏
4. **错误处理** - 正确的错误包装、不吞没错误
5. **资源管理** - 正确的清理（defer close）、context 传递
6. **性能** - 低效问题、内存泄漏

## .gitignore 支持

本工具完整支持 `.gitignore` 语法：

- 通配符模式：`*.pyc`, `*.log`
- 目录模式：`build/`, `dist/`
- 否定模式：`!important.py`, `!README.md`
- 双星号：`**/node_modules/**`
- 嵌套 .gitignore：自动扫描并应用所有子目录中的 .gitignore 规则

使用 `pathspec` 库实现，与 Git 行为完全一致。

## 环境变量

根据选择的 LLM 引擎，需要设置相应的环境变量：

### Claude
- `ANTHROPIC_API_KEY` - Anthropic API 密钥

### Gemini
- `GOOGLE_API_KEY` 或 `GEMINI_API_KEY` - Google API 密钥

## 依赖项

- `claude-agent-sdk` >= 0.1.18 - Claude Agent SDK
- `pathspec` >= 1.0.0 - Git 风格模式匹配
- `aiofiles` >= 24.1.0 - 异步文件 I/O
- `click` >= 8.3.1 - CLI 框架

## 项目结构

```
code-review-agent/
├── src/
│   └── code_review_agent/
│       ├── __init__.py      # 包初始化
│       ├── agent.py         # CodeReviewAgent - 主编排器
│       ├── cli.py           # CLI 入口点
│       ├── constants.py     # 配置常量
│       ├── gitignore.py     # GitignoreParser
│       ├── progress.py      # 进度显示
│       ├── state.py         # 状态管理（增量/续传）
│       └── llm/
│           ├── __init__.py  # LLM 工厂
│           ├── base.py      # LLM 抽象基类
│           ├── claude_agent.py  # Claude 实现
│           └── gemini_agent.py  # Gemini 实现
├── pyproject.toml           # 包配置
└── README.md
```

## License

MIT
