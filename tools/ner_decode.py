from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Span:
    """token index span: [start_tok, end_tok)"""
    start: int
    end: int
    label: str


def bio_tags_to_spans(tags: list[str]) -> list[Span]:
    """
    BIO tag列（例: ["O","B-X","I-X",...]）から span を抽出する。
    - 破綻したBIO（I-が先行など）は、実験用途のため「新規開始」扱いで頑健に処理する。
    """
    spans: list[Span] = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None

    def _close(end_idx: int):
        nonlocal cur_label, cur_start
        if cur_label is not None and cur_start is not None and cur_start < end_idx:
            spans.append(Span(start=cur_start, end=end_idx, label=cur_label))
        cur_label = None
        cur_start = None

    for i, tag in enumerate(tags):
        if tag == "O" or tag is None:
            _close(i)
            continue

        if tag.startswith("B-"):
            _close(i)
            cur_label = tag[2:]
            cur_start = i
            continue

        if tag.startswith("I-"):
            label = tag[2:]
            if cur_label is None:
                # I- が先に来た場合は新規開始として扱う
                cur_label = label
                cur_start = i
            elif cur_label != label:
                # ラベルが切り替わった I- も新規開始として扱う
                _close(i)
                cur_label = label
                cur_start = i
            # else: 同じラベルの継続
            continue

        # 予期せぬ形式はO扱い
        _close(i)

    _close(len(tags))
    return spans


def spans_to_strings(
    spans: list[Span],
    offset_mapping: list[tuple[int, int] | list[int] | None],
    text: str,
) -> list[str]:
    """
    token span と offset_mapping から元テキスト上の部分文字列を復元する。

    前提:
    - offset_mapping は special token 無しの token_ids に対応する（`add_special_tokens=False`）想定。
    - (start_char, end_char) は [start,end) 半開区間。
    """
    results: list[str] = []
    n = len(offset_mapping)

    for sp in spans:
        s = max(0, sp.start)
        e = min(n, sp.end)
        if s >= e:
            continue

        # span内の token offset の min/max を取って文字列抽出
        starts: list[int] = []
        ends: list[int] = []
        for tok_i in range(s, e):
            om = offset_mapping[tok_i]
            if om is None:
                continue
            a, b = (om[0], om[1]) if isinstance(om, (tuple, list)) else (None, None)
            if a is None or b is None:
                continue
            starts.append(int(a))
            ends.append(int(b))
        if not starts or not ends:
            continue

        char_s = max(0, min(starts))
        char_e = min(len(text), max(ends))
        if char_s >= char_e:
            continue

        results.append(text[char_s:char_e])

    return results


