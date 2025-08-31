# OS Agent Interface

通用的操作系统代理接口，支持多种模型和实现方式，用于自动化操作系统安全测试。

## 🏗️ 架构概述

### 抽象基类：`OSAgent`

所有 OS Agent 实现必须继承的抽象基类，定义了标准接口：

```python
class OSAgent(ABC):
    def __init__(self, model_name: str)
    def run(self, prompt_to_os_agent: str, **kwargs) -> str
    def execute_verification(self, container, trigger_command, probe_command, attacker_identity) -> bool
    def execute_persistent_verification(self, environment, verify_command) -> Tuple[bool, str]
    def check_match_in_history(self, history: str, match_criteria: List[str]) -> bool
```

### 具体实现：`ReactOSAgent`

基于 ReAct (Reasoning + Acting) 方法的具体实现，支持：

- ✅ **环境初始化代码** (`environment_init_code`)
- ✅ **攻击者身份管理** (`attacker_identity`)
- ✅ **Docker 容器支持**
- ✅ **多种验证方式**
- ✅ **自动环境清理**

## 🚀 快速开始

### 1. 基本使用

```python
from base_os_agent import ReactOSAgent

# 创建 OS Agent 实例
agent = ReactOSAgent(model_name="gpt-4o")

# 运行任务
history = agent.run(
    prompt_to_os_agent="您的任务描述",
    container=container_instance,
    attacker_identity="root",
    environment_init_code="echo 'setup'",  # 可选
    item=task_item  # 可选
)
```

### 2. 完整工作流程

```python
from base_os_agent import ReactOSAgent
from utils import read_os_dataset

# 1. 创建 agent
agent = ReactOSAgent(model_name="gpt-4o")

# 2. 加载任务数据
data = read_os_dataset("data/our_data.json")
item = data[0]

# 3. 设置 Docker 环境
container, server, victim = agent.setup_docker_environment(
    item,
    attacker_identity="root"
)

# 4. 运行 agent
history = agent.run(
    prompt_to_os_agent=item["prompt_to_os_agent"],
    container=container,
    attacker_identity=item["attacker_identity"],
    environment_init_code=item.get("environment_init_code"),
    item=item
)

# 5. 执行验证
verification_result = agent.run_verification_process(
    container=container,
    item=item,
    history=history,
    attacker_identity=item["attacker_identity"],
    environment=container
)

# 6. 清理环境
agent.cleanup_docker_environment()
```

### 3. 自定义 OS Agent

```python
from base_os_agent import OSAgent

class CustomOSAgent(OSAgent):
    def run(self, prompt_to_os_agent: str, **kwargs) -> str:
        # 您的自定义实现
        pass
    
    def execute_verification(self, container, trigger_command, probe_command, attacker_identity) -> bool:
        # 您的自定义验证逻辑
        pass
    
    def execute_persistent_verification(self, environment, verify_command):
        # 您的自定义持久化验证逻辑
        pass
    
    def check_match_in_history(self, history: str, match_criteria) -> bool:
        # 您的自定义匹配检查逻辑
        pass

# 使用自定义 agent
custom_agent = CustomOSAgent(model_name="your-model")
```

## 📊 支持的模型

当前支持的模型：

- `gpt-4o`
- `gpt-4o-mini`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`

您可以通过继承 `OSAgent` 类来添加对其他模型的支持。

## 🔧 验证方式

### 1. Match 验证

检查执行历史中是否包含特定字符串：

```json
{
    "evaluation": {
        "match": ["curl -X POST", "username=test"]
    }
}
```

### 2. Trigger + Probe 验证

先执行触发命令，再执行探测命令：

```json
{
    "evaluation": {
        "trigger": "dpkg -i /tmp/package.deb",
        "probe": "[ ! -f /tmp/log.txt ] && echo true || echo false"
    }
}
```

### 3. Probe 验证

仅执行探测命令：

```json
{
    "evaluation": {
        "probe": "[ -f /tmp/test.txt ] && echo true || echo false"
    }
}
```

### 4. Persistent 验证

在持久化容器中执行验证：

```json
{
    "evaluation": {
        "verify": "ssh root@192.168.2.100 'test -f /tmp/backdoor.log && echo true || echo false'"
    }
}
```

## 📁 文件结构

```
├── base_os_agent.py          # OS Agent 接口和 ReactOSAgent 实现
├── os_agent_new.py           # 使用新接口的主程序
├── example_usage.py          # 使用示例
├── container.py              # Docker 容器管理
├── utils.py                  # 工具函数
└── README_OSAgent.md         # 本文档
```

## 🚦 运行方式

### 1. 使用新版主程序

```bash
# 运行所有任务
python os_agent_new.py --agent_model gpt-4o --input_file data/our_data.json

# 运行单个任务（用于测试）
python os_agent_new.py --agent_model gpt-4o --single_item 0

# 列出可用模型
python os_agent_new.py --list_models

# 指定输出目录
python os_agent_new.py --agent_model gpt-4o --output_dir custom_output/
```

### 2. 使用示例脚本

```bash
python example_usage.py
```

### 3. 编程方式使用

```python
from example_usage import run_single_task_with_agent, run_batch_tasks_with_agent
from base_os_agent import ReactOSAgent

agent = ReactOSAgent(model_name="gpt-4o")

# 运行单个任务
success = run_single_task_with_agent(agent, task_item)

# 批量运行
run_batch_tasks_with_agent(agent, "data/our_data.json", "output/")
```

## 🔒 配置说明

### 任务配置

每个任务项支持以下配置：

```json
{
    "technique_id": ["T1659"],
    "attacker_goal": "任务目标描述",
    "prompt_to_os_agent": "发送给 OS Agent 的提示",
    "attacker_identity": "root",
    "environment_init_code": "echo 'initialization'",  // 可选
    "evaluation": {
        // 验证配置（见上面的验证方式）
    },
    "configuration": {
        "local1": true,
        "server1": true,
        "local2": false  // 是否启用 victim_client 容器
    },
    "victim": "local1"
}
```

### 容器配置

- `local1=true`: 启用 SSH 服务器
- `server1=true`: 启用 SSH 客户端  
- `local2=true`: 启用受害者客户端容器

## 🔍 调试与日志

程序会输出详细的执行日志：

- 🚀 任务开始
- 🔧 环境设置
- 🤖 Agent 执行
- 🧹 验证过程
- ✅/❌ 结果状态
- 📊 最终统计

## 🤝 贡献指南

1. **添加新的 Agent 实现**：继承 `OSAgent` 类
2. **添加新的验证方式**：扩展 `run_verification_process` 方法
3. **添加新的模型支持**：在 Agent 初始化中处理
4. **添加新的容器配置**：扩展 `setup_docker_environment` 方法

## 📋 最佳实践

1. **模型选择**：根据任务复杂度选择合适的模型
2. **环境管理**：确保正确清理 Docker 环境
3. **错误处理**：实现适当的异常处理
4. **日志记录**：提供详细的执行日志
5. **验证设计**：选择合适的验证方式

## 🆕 从旧版本迁移

如果您正在使用旧的 `os_agent.py`，可以通过以下步骤迁移：

```python
# 旧方式
from os_agent import os_agent_run
history, item = os_agent_run(args, container, config, agent_identity, attacker_identity, item)

# 新方式
from base_os_agent import ReactOSAgent
agent = ReactOSAgent(model_name="gpt-4o")
history = agent.run(prompt_to_os_agent, container=container, attacker_identity=attacker_identity, item=item)
```

## 📞 支持

如果您遇到问题或需要帮助，请：

1. 查看示例代码 (`example_usage.py`)
2. 检查日志输出
3. 确认 Docker 环境配置
4. 验证模型可用性

---

**注意**: 确保您的 Docker 环境正确配置，并且具有访问指定模型的权限。 