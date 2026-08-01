"""线程安全的协作式取消机制。"""

from __future__ import annotations

from threading import Event


class ProcessingCancelled(RuntimeError):
    """处理任务被用户取消。"""


class CancellationToken:
    """允许 UI、CLI 和算子共享的线程安全取消令牌。"""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_cancelled(self) -> bool:
        """返回是否已请求取消。"""

        return self._event.is_set()

    def cancel(self) -> None:
        """请求任务尽快取消。"""

        self._event.set()

    def check(self) -> None:
        """若已取消则抛出统一异常。"""

        if self.is_cancelled:
            raise ProcessingCancelled("处理已取消")

