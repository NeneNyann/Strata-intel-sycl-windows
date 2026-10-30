"""End-token ids and text held back in front of a stop string."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402
from serve import frontend, server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Detokenizer, Service, StopMatcher, end_token_ids, serve  # noqa: E402

QWEN_STOPS = ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "\nUser:"]
DEEPSEEK_STOPS = ["<｜end▁of▁sentence｜>", "<｜User｜>", "</s>", "\n\nHuman:"]
# words and endings that look like the start of a stop string or a piece of an end marker
WORDS = ["Nerazim", "Zim", "swim", "Neraz", "im", "im_end", "Nerazimville", "a|b|c|", "x > y and a < b", "the end of the text",
         "<", "<|", "<|im", "<|im_end", "|>", "friend", "legend", "\nUser", "\n\nHuman", "</", "</s", "<｜User", "<｜", "Nerazim<",
         "Nerazim|", "Nerazim <|im_start|"]






class EndTokenIds(unittest.TestCase):

    def test_qwen_markers_that_are_one_token_each_are_the_stop_ids(self):
        tok = ByteTokenizer()
        self.assertEqual(end_token_ids(tok), set(tok.encode("<|im_end|>", parse_special=True))
                         | set(tok.encode("<|endoftext|>", parse_special=True)))
        self.assertEqual(len(end_token_ids(tok)), 2)

    def test_no_marker_and_no_eos_means_no_stop_id(self):
        class Bare:
            def encode(self, text, parse_special=False):
                return list(text.encode())
        self.assertEqual(end_token_ids(Bare()), set())



class StopMatcherSplits(unittest.TestCase):
    """StopMatcher alone: every way to cut the text in two, and one character at a time."""

    @staticmethod
    def run_matcher(stops, chunks):
        m = StopMatcher(stops)
        out = "".join(m.push(c) for c in chunks)
        if m.hit is None:
            out += m.flush()
        return out, m.hit

    def chunkings(self, text):
        yield [text]
        for i in range(1, len(text)):
            yield [text[:i], text[i:]]
        yield list(text)

    def test_text_without_a_stop_string_comes_out_whole_at_every_split(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS, ["im_end"], ["im"], ["zim"], ["|", ">"], ["end"]):
            for word in WORDS:
                text = "My friend " + word
                if any(s in text for s in stops):
                    continue
                for chunks in self.chunkings(text):
                    with self.subTest(stops=stops, text=text, chunks=chunks):
                        self.assertEqual(self.run_matcher(stops, chunks), (text, None))

    def test_a_stop_string_cuts_exactly_there_at_every_split(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for stop in stops:
                for word in ("Nerazim", "swim"):
                    text = f"My friend {word}{stop}never seen"
                    for chunks in self.chunkings(text):
                        with self.subTest(stops=stops, text=text, chunks=chunks):
                            first = min((text.find(s), s) for s in stops if s in text)
                            self.assertEqual(self.run_matcher(stops, chunks), (text[:first[0]], first[1]))

    def test_the_held_back_tail_is_given_back_at_the_end(self):
        for stops in (QWEN_STOPS, DEEPSEEK_STOPS):
            for text in ("Nerazim <|im", "Nerazim <｜User", "Nerazim\nUser", "Nerazim <", "ends with a pipe |"):
                with self.subTest(stops=stops, text=text):
                    self.assertEqual(self.run_matcher(stops, list(text)), (text, None))

    def test_a_stop_string_that_starts_like_the_word_costs_nothing_but_the_wait(self):
        # "im_end" as a stop: "Nerazim" holds "im" until the next character says it is not the stop string
        m = StopMatcher(["im_end"])
        self.assertEqual(m.push("Neraz"), "Neraz")
        self.assertEqual(m.push("im"), "")
        self.assertEqual(m.held, "im")
        self.assertEqual(m.flush(), "im")






if __name__ == "__main__":
    unittest.main()
