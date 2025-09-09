"""
QlibProvider: 统一管理Qlib初始化、健康检查与可用性。
使用环境变量控制：
  QLIB_ENABLE=true|false
  QLIB_PROVIDER_URI=~/.qlib/qlib_data/cn_data
  QLIB_REGION=cn
若Qlib不可用，仍然允许系统运行（回退到Pandas/AKShare）。
"""

from __future__ import annotations

import os
import logging
from typing import Optional


class QlibProvider:
    _instance: Optional["QlibProvider"] = None

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.enabled = os.getenv("QLIB_ENABLE", "false").lower() == "true"
        self.provider_uri = os.getenv("QLIB_PROVIDER_URI", os.path.expanduser("~/.qlib/qlib_data/cn_data"))
        self.region = os.getenv("QLIB_REGION", "cn")
        self.available = False
        self.error: Optional[str] = None

        if not self.enabled:
            self.logger.info("QlibProvider disabled via env QLIB_ENABLE")
            return

        # 延迟导入，避免依赖冲突影响主进程启动
        try:
            import qlib  # type: ignore
            from qlib.config import REG_CN  # type: ignore

            region = REG_CN if self.region == "cn" else REG_CN  # 目前仅支持CN
            qlib.init(provider_uri=self.provider_uri, region=region)
            self.available = True
            self.logger.info(f"Qlib initialized: uri={self.provider_uri}, region={self.region}")
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.error = str(e)
            self.logger.warning(f"Qlib init failed, fallback to Pandas path. error={e}")

    @classmethod
    def get(cls) -> "QlibProvider":
        if cls._instance is None:
            cls._instance = QlibProvider()
        return cls._instance

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "provider_uri": self.provider_uri,
            "region": self.region,
            "error": self.error,
        }



