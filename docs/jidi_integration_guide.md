# 🔄 Jidi环境集成到ROLL框架指南

## 📋 概述

本文档详细记录了如何将Jidi平台的强化学习环境集成到ROLL训练框架的完整流程。通过这个指南，您可以轻松地将其他Jidi环境迁移到ROLL中进行LLM强化学习训练。

## 🎯 集成目标

- **保持原有功能**: 不修改Jidi原始代码，完全复用现有实现
- **ROLL兼容**: 符合ROLL框架的BaseEnv接口标准
- **文本交互**: 利用Jidi的LLM交互器实现文本输入输出
- **训练就绪**: 直接可用于ROLL的agentic训练流水线

---

## 🏗️ 架构设计

### 核心架构图

```
ROLL Framework
    └── agentic/env/
        └── jidi/                    # 新增Jidi环境包
            ├── __init__.py          # 包初始化
            ├── config.py            # 配置类
            └── env.py               # 环境实现

Jidi Project (外部依赖)
    ├── ai_lib/env/chooseenv.py      # 环境创建入口
    └── ai_lib/interactors/          # LLM交互器
```

### 设计原则

1. **包装器模式**: 在ROLL内部创建包装器，调用Jidi组件
2. **最小侵入**: 不修改Jidi代码，仅添加ROLL适配层
3. **接口兼容**: 严格遵循ROLL的BaseEnv接口
4. **错误处理**: 完善的异常处理和用户友好提示

---

## 🔧 实现步骤

### 步骤1: 创建目录结构

在ROLL项目中创建Jidi环境目录：

```bash
mkdir -p /path/to/ROLL/roll/agentic/env/jidi
```

### 步骤2: 实现配置类 (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Optional, List
from roll.agentic.env.base import BaseEnvConfig

@dataclass
class JidiBaseConfig(BaseEnvConfig):
    """Jidi环境基础配置类"""
    
    # Jidi特定配置
    jidi_env_name: str = field(default="", metadata={"help": "原始Jidi环境名称"})
    textualize: bool = field(default=True, metadata={"help": "启用文本交互"})
    jidi_project_path: str = field(default="/path/to/jidi", metadata={"help": "Jidi项目路径"})
    
    # ROLL标准配置
    max_steps: int = field(default=100)
    env_instruction: str = field(default="")
    action_pattern: str = field(default=r"<answer>(.*?)</answer>")
    max_tokens_per_step: int = field(default=128)
    
    # 错误处理配置
    format_penalty: float = field(default=-1.0)
    invalid_action_penalty: float = field(default=-0.1)
    
    def __post_init__(self):
        """生成默认指令"""
        if not self.env_instruction:
            self.env_instruction = self._get_default_instruction()
    
    def _get_default_instruction(self) -> str:
        """基于环境类型生成默认指令"""
        instructions = {
            "cliffwalking": "Navigate from start to goal while avoiding cliff...",
            "gobang_1v1": "Play Five in a Row game...",
            # 添加更多环境指令
        }
        return instructions.get(self.jidi_env_name, f"Play {self.jidi_env_name} game.")

@dataclass
class SpecificEnvConfig(JidiBaseConfig):
    """具体环境的配置类"""
    jidi_env_name: str = field(default="specific_env")
    max_steps: int = field(default=200)
    # 环境特定配置...
```

**🔑 关键点**:
- 继承 `BaseEnvConfig` 确保ROLL兼容性
- `jidi_env_name` 对应Jidi环境注册名
- `__post_init__` 用于生成默认值
- 为每个具体环境创建专门的配置类

### 步骤3: 实现环境类 (`env.py`)

```python
import sys
from typing import Any, Dict, List, Tuple
from roll.agentic.env.base import BaseEnv
from .config import JidiBaseConfig

class JidiBaseEnv(BaseEnv):
    """Jidi环境基类"""
    
    def __init__(self, config: JidiBaseConfig):
        super().__init__(config)
        self.config = config
        
        # 添加Jidi项目到Python路径
        if config.jidi_project_path not in sys.path:
            sys.path.insert(0, config.jidi_project_path)
        
        # 导入Jidi模块
        self._import_jidi_modules()
        
        # 初始化Jidi组件
        self._initialize_jidi_components()
        
        # 状态跟踪
        self._reset_internal_state()
    
    def _import_jidi_modules(self):
        """导入Jidi模块"""
        try:
            from ai_lib.env.chooseenv import make
            from ai_lib.interactors import LLM_INTERACTOR_REGISTRY
            self._make_env = make
            self._interactor_registry = LLM_INTERACTOR_REGISTRY
        except ImportError as e:
            raise ImportError(f"Failed to import Jidi modules: {e}")
    
    def _initialize_jidi_components(self):
        """初始化Jidi环境和交互器"""
        try:
            if self.config.textualize:
                # textualize=True时，make()返回交互器
                self.interactor = self._make_env(
                    self.config.jidi_env_name, 
                    textualize=True
                )
                self.jidi_env = self.interactor.game
            else:
                # textualize=False时，make()返回原始环境
                self.jidi_env = self._make_env(
                    self.config.jidi_env_name, 
                    textualize=False
                )
                # 手动创建交互器
                interactor_class = self._interactor_registry[self.config.jidi_env_name]
                self.interactor = interactor_class(self.jidi_env)
            
            self.n_player = getattr(self.interactor, 'n_player', 1)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Jidi components: {e}")
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """重置环境"""
        try:
            self._reset_internal_state()
            
            # 通过交互器重置
            text_obs = self.interactor.reset()
            
            # 处理单/多智能体格式
            if isinstance(text_obs, list):
                self._last_text_obs = text_obs[0] if text_obs else ""
            else:
                self._last_text_obs = text_obs if text_obs else ""
            
            # 创建信息字典
            info = {
                "env_name": self.config.jidi_env_name,
                "step_count": self._step_count,
                "max_steps": self.config.max_steps,
                "n_player": self.n_player,
                "episode_id": self._stats["episodes"]
            }
            
            return self._last_text_obs, info
            
        except Exception as e:
            error_msg = f"Reset failed: {str(e)}"
            error_info = {
                "error": True,
                "error_message": error_msg,
                "env_name": self.config.jidi_env_name,
                "step_count": 0,
                "max_steps": self.config.max_steps,
                "n_player": getattr(self, 'n_player', 1),
                "episode_id": self._stats.get("episodes", 0)
            }
            return f"Error: {error_msg}", error_info
    
    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict]:
        """执行动作"""
        self._step_count += 1
        self._stats["total_steps"] += 1
        
        info = {
            "step_count": self._step_count,
            "action_text": action,
            "action_valid": True,
            "action_successful": True,
            "error_message": None
        }
        
        try:
            # 通过交互器执行动作
            step_result = self.interactor.step(action)
            text_obs, reward, terminated, truncated, step_info = step_result
            
            # 处理返回格式
            if isinstance(text_obs, list):
                self._last_text_obs = text_obs[0] if text_obs else ""
            else:
                self._last_text_obs = text_obs if text_obs else ""
            
            if isinstance(reward, list):
                reward = reward[0] if reward else 0.0
            
            self._episode_reward += reward
            
            if isinstance(step_info, dict):
                info.update(step_info)
            
            # 检查最大步数
            if self._step_count >= self.config.max_steps:
                truncated = True
            
            self._stats["successful_actions"] += 1
            
        except Exception as e:
            # 错误处理
            error_msg = str(e)
            self._stats["failed_actions"] += 1
            
            if "parsing" in error_msg.lower() or "format" in error_msg.lower():
                reward = self.config.format_penalty
                self._stats["format_errors"] += 1
                info["format_error"] = True
            else:
                reward = self.config.invalid_action_penalty
                info["action_valid"] = False
            
            info.update({
                "action_successful": False,
                "error_message": error_msg,
                "penalty_applied": reward
            })
            
            truncated = self._step_count >= self.config.max_steps
        
        return self._last_text_obs, float(reward), terminated, truncated, info
    
    def render(self, mode: str = "text") -> Any:
        """渲染环境"""
        if mode == "text":
            return self._last_text_obs
        else:
            try:
                return self.jidi_env.render() if hasattr(self.jidi_env, 'render') else self._last_text_obs
            except Exception:
                return f"Render error"
    
    def get_all_actions(self) -> List[str]:
        """获取所有可用动作"""
        try:
            if hasattr(self.interactor, 'action_map'):
                return list(self.interactor.action_map.keys())
            else:
                return self._get_default_actions()
        except Exception:
            return self._get_default_actions()
    
    def _get_default_actions(self) -> List[str]:
        """环境特定的默认动作"""
        action_maps = {
            "cliffwalking": ["up", "down", "left", "right"],
            "gobang_1v1": ["coordinate_input"],
            # 添加更多环境的默认动作
        }
        return action_maps.get(self.config.jidi_env_name, ["action"])
    
    def _reset_internal_state(self):
        """重置内部状态"""
        self._step_count = 0
        self._episode_reward = 0.0
        self._last_text_obs = ""
        self._stats["episodes"] = self._stats.get("episodes", 0) + 1

class SpecificEnv(JidiBaseEnv):
    """具体环境实现"""
    
    def __init__(self, config=None):
        if config is None:
            config = SpecificEnvConfig()
        super().__init__(config)
    
    def reset(self, seed=None, **kwargs) -> Tuple[str, dict]:
        """带环境特定信息的重置"""
        text_obs, info = super().reset(seed=seed, **kwargs)
        
        # 添加环境特定信息
        info.update({
            "environment_type": "specific_type",
            # 其他环境特定信息...
        })
        
        return text_obs, info
```

**🔑 关键点**:
- `_initialize_jidi_components()` 正确处理Jidi的make()函数行为
- 错误处理要包含所有必需字段
- 状态跟踪确保统计信息准确
- 为具体环境创建专门的子类

### 步骤4: 注册环境

在 `roll/agentic/env/__init__.py` 中注册新环境：

```python
# 导入Jidi环境
from .jidi.config import SpecificEnvConfig
from .jidi.env import SpecificEnv

# 注册环境
REGISTERED_ENVS.update({
    "jidi_specific": SpecificEnv,
})

REGISTERED_ENV_CONFIGS.update({
    "jidi_specific": SpecificEnvConfig,
})
```

### 步骤5: 创建测试

创建 `tests/agentic/env/test_jidi_specific.py`：

```python
import pytest
from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS

def test_registration():
    """测试环境注册"""
    assert "jidi_specific" in REGISTERED_ENVS
    assert "jidi_specific" in REGISTERED_ENV_CONFIGS

def test_config():
    """测试配置创建"""
    config_class = REGISTERED_ENV_CONFIGS["jidi_specific"]
    config = config_class()
    assert config.jidi_env_name == "specific_env"
    assert len(config.env_instruction) > 0

def test_env_creation():
    """测试环境创建"""
    try:
        config_class = REGISTERED_ENV_CONFIGS["jidi_specific"]
        env_class = REGISTERED_ENVS["jidi_specific"]
        
        config = config_class(max_steps=50)
        env = env_class(config)
        
        assert env is not None
        assert hasattr(env, 'jidi_env')
        assert hasattr(env, 'interactor')
        
    except ImportError:
        pytest.skip("Jidi project not available")

def test_reset():
    """测试重置功能"""
    try:
        # 创建环境...
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert "env_name" in info
        assert info["env_name"] == "specific_env"
        
    except ImportError:
        pytest.skip("Jidi project not available")

def test_step():
    """测试步骤功能"""
    try:
        # 创建环境并重置...
        obs, reward, terminated, truncated, info = env.step("<answer>valid_action</answer>")
        
        assert isinstance(obs, str)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
    except ImportError:
        pytest.skip("Jidi project not available")

# 添加更多测试...
```

---

## 📚 环境特定适配指南

### 网格导航环境 (CliffWalking, GridWorld)

**特点**: 
- 离散动作空间 (up, down, left, right)
- 网格坐标系统
- 位置状态跟踪

**配置要点**:
```python
@dataclass
class NavigationConfig(JidiBaseConfig):
    grid_height: int = field(default=4)
    grid_width: int = field(default=12)
    step_penalty: float = field(default=-1.0)
```

### 棋类游戏 (Gobang, ChineseChess, Reversi)

**特点**:
- 多智能体交替行动
- 坐标或记谱法输入
- 复杂状态空间

**配置要点**:
```python
@dataclass
class BoardGameConfig(JidiBaseConfig):
    board_size: int = field(default=15)
    max_steps: int = field(default=225)  # 棋盘大小的1.5倍
    max_tokens_per_step: int = field(default=128)  # 更多token用于思考
```

**特殊处理**:
- 动作解析需要处理坐标格式或记谱法
- 多智能体需要正确处理轮流行动

### 实时游戏 (Snakes, Delivery)

**特点**:
- 连续动作执行
- 时间敏感
- 多智能体协作/竞争

**配置要点**:
```python
@dataclass
class RealtimeConfig(JidiBaseConfig):
    max_steps: int = field(default=300)
    max_tokens_per_step: int = field(default=64)  # 更少token提高响应速度
    action_timeout: float = field(default=1.0)
```

---

## ⚠️ 常见问题和解决方案

### 问题1: "too many values to unpack (expected 2)"

**原因**: Jidi的 `make(textualize=True)` 返回交互器而非环境

**解决方案**:
```python
if self.config.textualize:
    self.interactor = self._make_env(self.config.jidi_env_name, textualize=True)
    self.jidi_env = self.interactor.game
else:
    self.jidi_env = self._make_env(self.config.jidi_env_name, textualize=False)
    interactor_class = self._interactor_registry[self.config.jidi_env_name]
    self.interactor = interactor_class(self.jidi_env)
```

### 问题2: "Missing key env_name in info"

**原因**: 错误处理时没有包含所有必需字段

**解决方案**:
```python
except Exception as e:
    error_info = {
        "error": True,
        "error_message": str(e),
        "env_name": self.config.jidi_env_name,  # 确保包含
        "step_count": 0,
        "max_steps": self.config.max_steps,
        "n_player": getattr(self, 'n_player', 1),
        "episode_id": self._stats.get("episodes", 0)
    }
    return f"Error: {str(e)}", error_info
```

### 问题3: "'super' object has no attribute '__post_init__'"

**原因**: `BaseEnvConfig` 没有 `__post_init__` 方法

**解决方案**:
```python
def __post_init__(self):
    # 不要调用 super().__post_init__()
    if not self.env_instruction:
        self.env_instruction = self._get_default_instruction()
```

### 问题4: 动作解析失败

**原因**: Jidi交互器的动作格式与输入不匹配

**解决方案**:
- 检查环境的 `action_map` 或 `get_all_actions()` 方法
- 提供环境特定的默认动作列表
- 在错误处理中给出清晰的格式提示

---

## 🚀 训练配置示例

### ROLL配置文件示例

```yaml
# config.yaml
custom_envs:
  CliffWalking:
    env_type: "jidi_cliffwalking"
    env_config:
      max_steps: 200
      format_penalty: -2.0
      invalid_action_penalty: -0.5
    agent_system_template: |
      You are an intelligent agent navigating a cliff walking environment.
      Analyze the current state and choose the safest path to reach the goal.
      Always format your response as <answer>action</answer>.
    agent_template: |
      Current state:
      {observation}
      
      Choose your next action. Available actions: up, down, left, right.
      Response format: <answer>your_action</answer>
    reward_template: "Reward: {reward}"
    max_tokens_per_step: 64

train_env_manager:
  tags: ["CliffWalking"]
  num_groups_partition: [128]
  group_size: 1
  max_traj_per_env: 10
```

### Python训练脚本示例

```python
from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS

def main():
    # 创建配置
    config_class = REGISTERED_ENV_CONFIGS["jidi_cliffwalking"]
    config = config_class(
        max_steps=100,
        format_penalty=-1.0,
        env_instruction="Navigate safely to the goal..."
    )
    
    # 创建环境
    env_class = REGISTERED_ENVS["jidi_cliffwalking"]
    env = env_class(config)
    
    # 开始训练...
    obs, info = env.reset(seed=42)
    
    for step in range(config.max_steps):
        # LLM生成动作
        action = llm_generate_action(obs)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
```

---

## 📋 检查清单

在完成新环境集成后，使用此清单验证：

### ✅ 代码实现
- [ ] 配置类继承 `BaseEnvConfig`
- [ ] 环境类继承 `BaseEnv` 
- [ ] 正确处理Jidi的 `make()` 函数
- [ ] 实现所有必需方法 (`reset`, `step`, `render`, `get_all_actions`)
- [ ] 错误处理包含所有必需字段
- [ ] 支持单/多智能体格式

### ✅ 注册和配置
- [ ] 在 `REGISTERED_ENVS` 中注册环境类
- [ ] 在 `REGISTERED_ENV_CONFIGS` 中注册配置类
- [ ] 环境名称使用 `jidi_` 前缀
- [ ] 配置参数合理设置

### ✅ 测试验证
- [ ] 注册测试通过
- [ ] 配置创建测试通过
- [ ] 环境创建测试通过
- [ ] reset()功能测试通过
- [ ] step()功能测试通过
- [ ] 错误处理测试通过
- [ ] 完整回合测试通过

### ✅ 文档和示例
- [ ] 添加环境特定的配置说明
- [ ] 创建使用示例
- [ ] 更新环境列表文档
- [ ] 记录特殊注意事项

---

## 🔮 未来扩展

### 计划中的环境

1. **GridWorld** - 网格世界导航
2. **Gobang** - 五子棋对弈
3. **ChineseChess** - 中国象棋
4. **Reversi** - 黑白棋
5. **Snakes** - 贪吃蛇竞技
6. **Sokoban** - 推箱子益智
7. **Delivery** - 多智能体配送
8. **Seek** - 捉迷藏游戏
9. **MiniGrid** - 迷你网格世界

### 高级特性

- **视觉环境支持**: 集成VLM交互器
- **多模态输入**: 支持图像+文本输入
- **自定义奖励**: 可配置的奖励函数
- **实时可视化**: 训练过程可视化
- **性能优化**: 并行环境执行

---

## 📞 支持和反馈

如果在集成过程中遇到问题：

1. **检查日志**: 查看详细错误信息
2. **验证路径**: 确保Jidi项目路径正确
3. **测试依赖**: 确认Jidi环境可单独运行
4. **参考示例**: 查看CliffWalking的完整实现
5. **逐步调试**: 使用调试脚本逐步验证

---

## 📄 附录

### A. 完整文件结构

```
ROLL/
├── roll/agentic/env/
│   ├── __init__.py                 # 更新注册表
│   └── jidi/
│       ├── __init__.py             # 包初始化
│       ├── config.py               # 配置类
│       └── env.py                  # 环境实现
├── tests/agentic/env/
│   └── test_jidi_specific.py       # 测试文件
└── examples/
    └── jidi_specific_example.py    # 使用示例
```

### B. 命名约定

- **环境名称**: `jidi_{original_name}` (如 `jidi_cliffwalking`)
- **配置类**: `{EnvName}Config` (如 `CliffWalkingConfig`)
- **环境类**: `{EnvName}Env` (如 `CliffWalkingEnv`)
- **测试文件**: `test_jidi_{env_name}.py`

### C. 参考资源

- [ROLL框架文档](https://github.com/volcengine/ROLL)
- [Jidi平台环境](https://github.com/jidiai/ai_lib)
- [Gymnasium标准](https://gymnasium.farama.org/)

---

*最后更新: 2025-08-18*
