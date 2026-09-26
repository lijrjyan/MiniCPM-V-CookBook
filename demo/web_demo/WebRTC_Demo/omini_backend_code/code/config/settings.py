"""
配置管理模块 - 基于Pydantic Settings + YAML
支持多环境配置（dev/prod/test），通过APP_ENV环境变量激活
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    """Redis配置"""
    host: str = Field(default="localhost", description="Redis主机地址")
    port: int = Field(default=6379, description="Redis端口")
    db: int = Field(default=0, description="Redis数据库")
    password: Optional[str] = Field(default=None, description="Redis密码")
    username: Optional[str] = Field(default=None, description="Redis用户名")
    max_connections: int = Field(default=20, description="最大连接数")
    ssl: bool = Field(default=False, description="是否启用SSL")
    socket_timeout: float = Field(default=5.0, description="套接字超时（秒）")
    socket_connect_timeout: float = Field(default=5.0, description="连接超时（秒）")

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        case_sensitive=False
    )


class LiveKitSettings(BaseSettings):
    """LiveKit配置"""
    url: str = Field(default="", description="LiveKit服务器URL")
    api_key: str = Field(default="", description="LiveKit API Key")
    api_secret: str = Field(default="", description="LiveKit API Secret")

    model_config = SettingsConfigDict(
        env_prefix="LIVEKIT_",
        case_sensitive=False
    )


class ServerSettings(BaseSettings):
    """服务器配置"""
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8021, description="监听端口")
    debug: bool = Field(default=False, description="调试模式")
    reload: bool = Field(default=False, description="自动重载")
    workers: int = Field(default=1, description="工作进程数")
    data_root_dir: str = Field(default="/cache/zhangtao/intput", description="数据根目录")
    tts_bin_dir: str = Field(default="/cache/caitianchi/temp/o45_cpp_stable/output/tts_bin", description="TTS bin目录")
    
    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        case_sensitive=False
    )


class LoggingSettings(BaseSettings):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    enable_console: bool = Field(default=True, description="启用控制台日志")
    enable_file: bool = Field(default=True, description="启用文件日志")
    enable_unified_file: bool = Field(default=True, description="启用统一日志文件")
    max_file_size: int = Field(default=20 * 1024 * 1024, description="最大日志文件大小（字节）")
    backup_count: int = Field(default=20, description="日志备份数量")
    log_dir: str = Field(default="logs", description="日志目录")

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class CORSSettings(BaseSettings):
    """CORS配置"""
    allow_origins: List[str] = Field(default=["*"], description="允许的源")
    allow_credentials: bool = Field(default=True, description="允许凭证")
    allow_methods: List[str] = Field(default=["*"], description="允许的方法")
    allow_headers: List[str] = Field(default=["*"], description="允许的头")

    model_config = SettingsConfigDict(
        env_prefix="CORS_",
        case_sensitive=False
    )


class HeartbeatSettings(BaseSettings):
    """心跳监控配置"""
    lock_key: str = Field(default="inference:heartbeat:lock", description="心跳监控分布式锁键")
    lock_timeout: int = Field(default=20, description="锁超时时间（秒）")
    monitoring_interval: int = Field(default=10, description="监控间隔（秒）")
    cleanup_interval: int = Field(default=20, description="清理间隔（秒）")

    model_config = SettingsConfigDict(
        env_prefix="HEARTBEAT_",
        case_sensitive=False
    )


class InferenceServiceSettings(BaseSettings):
    """推理服务管理配置"""
    services_key: str = Field(default="inference:services", description="Redis中存储服务信息的Hash键")
    heartbeat_timeout: int = Field(default=20, description="心跳超时时间（秒）")
    lock_timeout: int = Field(default=200, description="服务锁定超时时间（秒）")

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_SERVICE_",
        case_sensitive=False
    )


class VoiceChatSettings(BaseSettings):
    """语音聊天配置"""
    enable_voice_interruption: bool = Field(default=False, description="是否启用语音打断功能")
    voice_interruption_threshold: float = Field(default=0.85, description="语音打断阈值（0-1）")

    model_config = SettingsConfigDict(
        env_prefix="VOICE_CHAT_",
        case_sensitive=False
    )


class Settings(BaseSettings):
    """
    应用配置主类
    通过APP_ENV环境变量激活不同的配置文件（dev/prod/test/local）
    配置优先级：环境变量 > 环境配置文件 > base配置文件 > 默认值
    """
    # 应用基础信息
    app_name: str = Field(default="MiniCPMO Web Backend", description="应用名称")
    app_version: str = Field(default="0.1.0", description="应用版本")
    app_env: str = Field(default="local", description="应用环境（dev/prod/test/local）")
    
    # 子配置
    redis: RedisSettings = Field(default_factory=RedisSettings)
    livekit: LiveKitSettings = Field(default_factory=LiveKitSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    heartbeat: HeartbeatSettings = Field(default_factory=HeartbeatSettings)
    inference_service: InferenceServiceSettings = Field(default_factory=InferenceServiceSettings)
    voice_chat: VoiceChatSettings = Field(default_factory=VoiceChatSettings)

    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",  # 支持嵌套配置，如 REDIS__HOST
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        # from_yaml passes the YAML as init kwargs, which pydantic-settings ranks above the environment by default;
        # put the environment first so SERVER__PORT, LIVEKIT__URL, ... really override the YAML as documented above.
        return env_settings, init_settings, dotenv_settings, file_secret_settings

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """验证环境名称"""
        valid_envs = ["dev", "prod", "test", "local"]
        v_lower = v.lower()
        if v_lower not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v_lower

    @classmethod
    def from_yaml(cls, config_dir: Optional[Path] = None) -> "Settings":
        """
        从YAML文件加载配置
        配置加载顺序：
        1. 加载 base.yaml（基础配置）
        2. 加载 {app_env}.yaml（环境特定配置）
        3. 环境变量覆盖（最高优先级）
        """
        if config_dir is None:
            # 默认配置目录为项目根目录下的config
            config_dir = Path(__file__).parent

        # 获取环境变量
        app_env = os.getenv("APP_ENV", "dev").lower()
        
        # 加载base配置
        base_config_path = config_dir / "base.yaml"
        config_data = {}
        
        if base_config_path.exists():
            with open(base_config_path, "r", encoding="utf-8") as f:
                base_data = yaml.safe_load(f) or {}
                config_data = cls._merge_dict(config_data, base_data)
        
        # 加载环境特定配置
        env_config_path = config_dir / f"{app_env}.yaml"
        if env_config_path.exists():
            with open(env_config_path, "r", encoding="utf-8") as f:
                env_data = yaml.safe_load(f) or {}
                config_data = cls._merge_dict(config_data, env_data)
        
        # 确保app_env设置正确
        config_data["app_env"] = app_env
        
        # 创建Settings实例（环境变量会自动覆盖）
        return cls(**config_data)

    @staticmethod
    def _merge_dict(base: Dict[Any, Any], override: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        深度合并字典
        override中的值会覆盖base中的值
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env,
            "redis": self.redis.model_dump(),
            "livekit": {
                **self.livekit.model_dump(exclude={"api_secret"}),
                "api_secret": "***" if self.livekit.api_secret else ""
            },
            "server": self.server.model_dump(),
            "logging": self.logging.model_dump(),
            "cors": self.cors.model_dump(),
            "heartbeat": self.heartbeat.model_dump(),
            "inference_service": self.inference_service.model_dump(),
            "voice_chat": self.voice_chat.model_dump(),
        }


# 全局配置实例缓存
@lru_cache()
def get_settings() -> Settings:
    """
    获取配置实例（单例模式）
    配置加载顺序：
    1. YAML文件（base.yaml + {env}.yaml）
    2. 环境变量覆盖
    """
    return Settings.from_yaml()


# 便捷访问
def get_redis_settings() -> RedisSettings:
    """获取Redis配置"""
    return get_settings().redis


def get_livekit_settings() -> LiveKitSettings:
    """获取LiveKit配置"""
    return get_settings().livekit


def get_server_settings() -> ServerSettings:
    """获取服务器配置"""
    return get_settings().server


def get_logging_settings() -> LoggingSettings:
    """获取日志配置"""
    return get_settings().logging


def get_cors_settings() -> CORSSettings:
    """获取CORS配置"""
    return get_settings().cors


def get_heartbeat_settings() -> HeartbeatSettings:
    """获取心跳监控配置"""
    return get_settings().heartbeat


def get_inference_service_settings() -> InferenceServiceSettings:
    """获取推理服务管理配置"""
    return get_settings().inference_service


def get_voice_chat_settings() -> VoiceChatSettings:
    """获取语音聊天配置"""
    return get_settings().voice_chat

