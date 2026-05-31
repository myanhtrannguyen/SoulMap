import os
import itertools
from typing import List, Optional


class KeyManager:
    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            raw = os.getenv("GEMINI_API_KEYS", "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]

        if not keys:
            raise RuntimeError("No Gemini API keys provided")

        self.keys = keys
        self._cycle = itertools.cycle(self.keys)

    def next(self) -> str:
        return next(self._cycle)