import logging
import sys


class DebugLogger:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._setup_logger()

    def _setup_logger(self):
        """配置日志格式"""
        logging.basicConfig(
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format="🪵 [%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def set_debug(self, debug_mode: bool):
        """动态设置调试模式"""
        self.debug_mode = debug_mode
        level = logging.DEBUG if debug_mode else logging.INFO
        self.logger.setLevel(level)

    def info(self, message: str):
        """普通信息日志"""
        self.logger.info(f"ℹ️  {message}")

    def debug(self, message: str):
        """调试信息，只在debug模式输出"""
        if self.debug_mode:
            self.logger.debug(f"🐛 {message}")

    def warning(self, message: str):
        """警告信息"""
        self.logger.warning(f"⚠️  {message}")

    def error(self, message: str):
        """错误信息"""
        self.logger.error(f"❌ {message}")

    def success(self, message: str):
        """成功信息"""
        self.logger.info(f"✅ {message}")
