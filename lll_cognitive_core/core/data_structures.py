from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from openai import OpenAI
from lll_simple_ai_shared import UnderstoodData, EpisodicMemoriesModels
from ..config.create_openai_config import CreateOpenaiConfig


class UnderstandEventData(BaseModel):
    type: str
    data: str
    source: str
    timestamp: datetime


class UnderstandEventInput(BaseModel):
    understand_event: UnderstandEventData
    recent_events: List["CognitiveEvent"]
    active_goals: List["Goal"]


class AssociativeRecallInput(BaseModel):
    current_situation: str
    main_events: str
    recent_events: List["CognitiveEvent"]
    episodic_memories: List["EpisodicMemoriesModels"]
    query_too_many_results: bool
    active_goals: List["Goal"]


class GenerateBehaviorInput(BaseModel):
    current_situation: str
    main_events: str
    recent_events: List["CognitiveEvent"]
    episodic_memories: List["EpisodicMemoriesModels"]
    episodic_memories_text: str | None
    active_goals: List["Goal"]
    social_norms: List[str]


class ExtractMemoriesInput(BaseModel):
    current_situation: str
    recent_events: List["CognitiveEvent"]
    active_goals: List["Goal"]


@dataclass
class WorkingMemory:
    # 当前活跃信息
    current_situation: str  # 当前情境理解
    active_goals: List["Goal"]  # 活跃目标
    # attention_focus: Optional[str]  # 当前注意力焦点

    # 短期事件缓存
    recent_events: List["CognitiveEvent"]  # 最近事件(循环队列，最大50个)
    # event_buffer: List["CognitiveEvent"]  # 待处理事件缓冲区

    # 上下文状态
    # social_context: "SocialContext"  # 社交上下文

    # 元信息
    cognitive_load: float  # 当前认知负荷 0-1
    last_update_time: float  # 最后更新时间戳
    active_duration: float  # 当前会话持续时间


@dataclass
class ExtractMemories:
    current_situation: str  # 当前情境理解
    active_goals: List["Goal"]  # 活跃目标
    recent_events: List["CognitiveEvent"]  # 最近事件
    social_context: "SocialContext"


@dataclass
class CognitiveEvent:
    event_id: str  # 事件唯一ID
    timestamp: float  # 发生时间戳
    source: str  # 事件来源
    # event_type: str  # 事件类型
    modality_type: str  # asr 语音识别输入 tts 语音输出 motion 动作执行 vision 图像识别 sensor 传感器数据 system 系统状态
    raw_data: UnderstandEventData  # 原始数据
    understood_data: UnderstoodData  # 理解后的数据
    importance_score: float  # 重要性评分
    # tags: List[str]  # 标签
    # spatial_info: Optional["SpatialInfo"]  # 空间信息


# 基础枚举类型


class UrgencyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmotionalState(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    ANGRY = "angry"
    CONFUSED = "confused"
    CURIOUS = "curious"


class CoreStatus(Enum):
    AWAITING = "awaiting"
    AWARE = "aware"
    WINDING_DOWN = "winding_down"
    DREAMING = "dreaming"


@dataclass
class EpisodicMemory:
    episodic_memories: Dict[str, EpisodicMemoriesModels]  # 记忆片段列表
    keyword_index: Dict[str, List[str]]  # 关键词索引
    association_index: Dict[str, List[str]]  # 联想词索引
    time_index: Dict[str, List[str]]  # 时间索引


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any]  # 用户偏好
    behavior_patterns: List["BehaviorPattern"]  # 行为模式
    interaction_style: str  # 交互风格
    trust_level: float  # 信任程度
    last_interaction: float  # 最后交互时间


@dataclass
class Fact:
    fact_id: str
    statement: str  # 事实陈述
    confidence: float
    supporting_evidence: List[str]  # 支持证据
    last_verified: float  # 最后验证时间


@dataclass
class SocialContext:
    participants: List["Participant"]
    social_norms: Dict[str, Any]  # 社交规范
    relationship_context: Dict[str, float]  # 关系上下文
    conversation_history: List["ConversationTurn"]
    social_roles: Dict[str, str]  # 社交角色


@dataclass
class Participant:
    participant_id: str
    role: str
    attention_level: float
    engagement: float
    last_interaction: float


@dataclass
class EventPrimaryDataContext:
    text: Optional[str]
    audio_features: Optional[Dict]  # 音频特征
    image_data: Optional[Dict]  # 图像数据或特征
    detected_objects: Optional[List]  # 检测到的物体
    joint_states: Optional[Dict]  # 关节状态
    motion_trajectory: Optional[Dict]  # 运动轨迹
    sensor_readings: Optional[Dict]  # 传感器读数


@dataclass
class Goal:
    goal_id: str
    goal_type: str  # 目标类型
    description: str  # 目标描述
    priority: int  # 优先级
    status: str  # 状态
    subgoals: List["Goal"]  # 子目标
    constraints: List[str]  # 约束条件
    success_criteria: List[str]  # 成功标准
    created_time: float
    deadline: Optional[float]  # 截止时间


@dataclass
class DefaultPluginInitOptions:
    client: OpenAI
    config: CreateOpenaiConfig
    task_pre_messages: Optional[List[Dict[str, str]]] = None
