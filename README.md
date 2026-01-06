# Code Review Agent

<p align="center">

**智能代码审查工具** - 基于 Claude Agent SDK

[English](#english) | [中文](#中文)

</p>

---

<a name="中文"></a>

## 中文

一个强大的智能代码审查工具，使用 Claude Agent SDK 构建，支持多种编程语言，具有代码库探索、并行执行、分块审查等特性。

### 主要特性

- **代码库探索** - 自动分析代码库，生成项目特定的审查规则
- **智能 .gitignore 支持** - 使用 `pathspec` 库完整支持 .gitignore 语法，包括嵌套 .gitignore 文件
- **并行审查** - 多文件并发审查，提升效率
- **大文件分块处理** - 自动将大文件分割为可管理的代码块进行审查
- **增量保存** - 每个文件审查完成后立即保存，支持断点续传
- **中文支持** - 审查报告支持中文输出
- **详细报告** - 生成单独的文件审查和汇总报告

### 安装

```bash
# 克隆仓库
git clone https://github.com/zhuwenxing/code-review-agent.git
cd code-review-agent

# 创建虚拟环境并安装依赖
uv venv -p 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 使用方法

#### 基本用法

```bash
# 审查当前目录的代码
python code_review_agent.py .

# 审查指定目录
python code_review_agent.py /path/to/your/code
```

#### 高级选项

```bash
python code_review_agent.py /path/to/code \
  --extensions py,go,js,ts \
  --output-dir reviews \
  --concurrency 10 \
  --max-files 100 \
  --skip-explore
```

#### 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | - | - | 要审查的目录路径（必需） |
| `--extensions` | `-e` | `py,go,js,ts,java,cpp,c,h` | 要审查的文件扩展名 |
| `--output-dir` | `-o` | `reviews` | 审查报告输出目录 |
| `--max-files` | `-m` | `None` | 最大审查文件数量 |
| `--concurrency` | `-c` | `5` | 并发审查工作线程数 |
| `--retry` | `-r` | `2` | 失败重试次数 |
| `--chunk-lines` | - | `500` | 大文件分块的行数 |
| `--large-file-threshold` | - | `800` | 触发分块审查的行数阈值 |
| `--skip-explore` | - | `False` | 跳过代码库探索阶段 |

### 输出结构

```
reviews/
├── SUMMARY.md                    # 汇总报告（含统计数据）
├── src_main.py.md                # 单个文件的审查报告
├── src_utils_helper.go.md        # 其他文件的审查报告
└── ...
```

### 审查流程

工具分 4 个阶段运行：

1. **阶段 1：探索** - 分析代码库以了解模式并生成特定的审查规则
2. **阶段 2：发现** - 查找所有匹配指定扩展名的文件（应用 .gitignore 过滤）
3. **阶段 3：审查** - 并行审查文件，并增量保存结果
4. **阶段 4：汇总** - 生成最终汇总报告

### 标准审查领域

工具检查以下方面：

1. **安全性** - 漏洞、注入风险
2. **缺陷** - 潜在 bug、边界情况、nil/null 问题
3. **并发** - 竞态条件、死锁、goroutine 泄漏
4. **错误处理** - 正确的错误包装、不吞没错误
5. **资源管理** - 正确的清理（defer close）、context 传递
6. **性能** - 低效问题、内存泄漏

### .gitignore 支持

本工具完整支持 `.gitignore` 语法：

- ✅ 通配符模式：`*.pyc`, `*.log`
- ✅ 目录模式：`build/`, `dist/`
- ✅ 否定模式：`!important.py`, `!README.md`
- ✅ 双星号：`**/node_modules/**`
- ✅ 嵌套 .gitignore：自动扫描并应用所有子目录中的 .gitignore 规则

使用 `pathspec` 库实现，与 Git 行为完全一致。

### 示例输出

#### 单个文件审查

```markdown
# 代码审查：src/main.py

**审查时间**：2026-01-06T10:30:00
**行数**：150
**分块**：否
**状态**：completed

---

- **评分**：4/5
- **总结**：结构良好，错误处理得当
- **问题**：
  - 第 45 行：数据库连接缺少错误处理
  - 第 78 行：错误路径上资源未正确关闭
- **建议**：
  - 为数据库连接添加上下文管理器
  - 考虑对 I/O 操作使用 async/await
```

#### 汇总报告

```markdown
# 代码审查汇总报告

**生成时间**：2026-01-06 10:35:00
**审查文件数**：25
**吞吐量**：2.5 文件/秒

## 执行摘要
[AI 生成的常见问题和建议摘要]

## 统计信息
| 指标 | 值 |
|------|-----|
| 总文件数 | 25 |
| 成功审查 | 24 |
| 错误 | 1 |
```

### 依赖项

- `claude-agent-sdk` >= 0.1.18
- `pathspec` >= 1.0.0

### License

MIT

---

<a name="english"></a>

## English

An intelligent code review agent powered by Claude Agent SDK that performs automated code reviews with parallel execution, chunked review for large files, and codebase-aware rule generation.

### Features

- **Codebase Exploration**: Automatically analyzes your codebase to generate project-specific review rules
- **Smart .gitignore Support**: Full .gitignore syntax support using `pathspec` library, including nested .gitignore files
- **Parallel Execution**: Reviews multiple files concurrently for faster results
- **Chunked Review**: Handles large files by splitting them into manageable chunks
- **Incremental Save**: Each file review is saved immediately as it completes
- **Chinese Language Support**: Review reports support Chinese output
- **Comprehensive Reports**: Generates individual file reviews and a summary report

### Installation

```bash
# Clone the repository
git clone https://github.com/zhuwenxing/code-review-agent.git
cd code-review-agent

# Create virtual environment and install dependencies
uv venv -p 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Usage

#### Basic Usage

```bash
# Review current directory
python code_review_agent.py .

# Review specific directory
python code_review_agent.py /path/to/your/code
```

#### Advanced Options

```bash
python code_review_agent.py /path/to/code \
  --extensions py,go,js,ts \
  --output-dir reviews \
  --concurrency 10 \
  --max-files 100 \
  --skip-explore
```

#### Command-line Arguments

| Parameter | Short | Default | Description |
|----------|-------|---------|-------------|
| `path` | - | - | Path to the directory to review (required) |
| `--extensions` | `-e` | `py,go,js,ts,java,cpp,c,h` | File extensions to review |
| `--output-dir` | `-o` | `reviews` | Output directory for reviews |
| `--max-files` | `-m` | `None` | Maximum number of files to review |
| `--concurrency` | `-c` | `5` | Number of concurrent review workers |
| `--retry` | `-r` | `2` | Number of retries on failure |
| `--chunk-lines` | - | `500` | Lines per chunk for large files |
| `--large-file-threshold` | - | `800` | Line threshold for chunked review |
| `--skip-explore` | - | `False` | Skip codebase exploration phase |

#### Output Structure

```
reviews/
├── SUMMARY.md                    # Summary report with statistics
├── src_main.py.md                # Individual file reviews
├── src_utils_helper.go.md
└── ...
```

#### Review Process

The agent runs in 4 phases:

1. **Phase 1: Exploration** - Analyzes codebase to understand patterns and generate specific review rules
2. **Phase 2: Discovery** - Finds all files matching the specified extensions (with .gitignore filtering)
3. **Phase 3: Review** - Reviews files in parallel with incremental saves
4. **Phase 4: Summary** - Generates final summary report

#### Standard Review Areas

The agent checks for:

1. **Security**: Vulnerabilities, injection risks
2. **Bugs**: Potential bugs, edge cases, nil/null issues
3. **Concurrency**: Race conditions, deadlocks, goroutine leaks
4. **Error Handling**: Proper error wrapping, not swallowing errors
5. **Resources**: Proper cleanup (defer close), context passing
6. **Performance**: Inefficiencies, memory leaks

#### .gitignore Support

This tool has full `.gitignore` syntax support:

- ✅ Wildcard patterns: `*.pyc`, `*.log`
- ✅ Directory patterns: `build/`, `dist/`
- ✅ Negation patterns: `!important.py`, `!README.md`
- ✅ Double-star patterns: `**/node_modules/**`
- ✅ Nested .gitignore: Automatically scans and applies all .gitignore files in subdirectories

Implemented using `pathspec` library, matching Git's behavior exactly.

#### Example Output

##### Individual File Review

```markdown
# Code Review: src/main.py

**Reviewed at**: 2026-01-06T10:30:00
**Lines**: 150
**Chunked**: No
**Status**: completed

---

- **Score**: 4/5
- **Summary**: Well-structured code with good error handling
- **Issues**:
  - Line 45: Missing error handling for database connection
  - Line 78: Resource not properly closed on error path
- **Recommendations**:
  - Add context manager for database connection
  - Consider using async/await for I/O operations
```

##### Summary Report

```markdown
# Code Review Summary Report

**Generated**: 2026-01-06 10:35:00
**Files Reviewed**: 25
**Throughput**: 2.5 files/sec

## Executive Summary
[AI-generated summary of common issues and recommendations]

## Statistics
| Metric | Value |
|--------|-------|
| Total Files | 25 |
| Successfully Reviewed | 24 |
| Errors | 1 |
```

### Dependencies

- `claude-agent-sdk` >= 0.1.18
- `pathspec` >= 1.0.0

### License

MIT
