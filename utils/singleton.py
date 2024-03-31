from typing import Any
import threading


class singleton(type):
    instances_ = {}
    lock_ = threading.Lock()
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        with cls.lock_:
            if cls not in cls.instances_:
                cls.instances_[cls] = super(singleton, cls).__call__(*args, **kwds)
        return cls.instances_[cls]