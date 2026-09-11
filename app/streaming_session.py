from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewSnapshot:
    committed: str
    tail: str
    state: str = "recording"

    @property
    def visible_text(self) -> str:
        return f"{self.committed}{self.tail}"


class PreviewAssembler:
    def __init__(self, context_chars: int = 30) -> None:
        self._context_chars = context_chars
        self._committed = ""
        self._tail = ""
        self._latest_sequence = 0
        self._llm_sequence = 0

    def snapshot(self, state: str = "recording") -> PreviewSnapshot:
        return PreviewSnapshot(self._committed, self._tail, state=state)

    def apply_asr_segment(self, sequence: int, text: str) -> PreviewSnapshot:
        if sequence < self._latest_sequence:
            return self.snapshot()
        self._latest_sequence = sequence
        self._llm_sequence = sequence
        self._tail = self._merge_tail(self._tail, text)
        return self.snapshot()

    def apply_llm_update(self, sequence: int, text: str) -> PreviewSnapshot:
        if sequence != self._llm_sequence:
            return self.snapshot()
        self._tail = text
        return self.snapshot(state="processing")

    def promote_tail(self) -> PreviewSnapshot:
        self._committed += self._tail
        self._tail = ""
        return self.snapshot()

    @staticmethod
    def _merge_tail(existing_tail: str, new_text: str) -> str:
        max_overlap = min(len(existing_tail), len(new_text))
        for size in range(max_overlap, 0, -1):
            if existing_tail[-size:] == new_text[:size]:
                return existing_tail + new_text[size:]
        return new_text
