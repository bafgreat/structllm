from __future__ import annotations

import os
import unittest
from pathlib import Path

from structllm.chat_format import parse_generation, split_reasoning, to_template_messages

TOKENIZERS = Path(os.environ.get("STRUCTLLM_TEST_TOKENIZERS", "/Volumes/X9/scratch/structllm/hf-meta"))

CALL = {
    "id": "call_1",
    "name": "build_structure",
    "arguments": {
        "project_id": "11111111-1111-4111-8111-111111111111",
        "parameters": {"body": {"smiles": "OC(=O)c1ccc(cc1)C(=O)O", "name": "terephthalic acid"}},
    },
}
MESSAGES = [
    {"role": "system", "content": "You are a test assistant."},
    {"role": "user", "content": "Build terephthalic acid."},
    {"role": "assistant", "content": "", "reasoning": "Need a SMILES build.", "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "call_1", "name": "build_structure", "content": '{"status":"succeeded"}'},
    {"role": "assistant", "content": "Built.", "reasoning": "Report it.", "tool_calls": []},
]


def load_tokenizer(name: str):
    path = TOKENIZERS / name
    if not path.exists():
        raise unittest.SkipTest(f"tokenizer files not available at {path}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # train extra not installed
        raise unittest.SkipTest(str(error))
    return AutoTokenizer.from_pretrained(path)


class ParserTests(unittest.TestCase):
    def test_reasoning_split_with_and_without_open_tag(self) -> None:
        self.assertEqual(split_reasoning("<think>\nplan\n</think>\n\nanswer"), ("plan", "answer"))
        self.assertEqual(split_reasoning("plan only\n</think>\nanswer"), ("plan only", "answer"))
        self.assertEqual(split_reasoning("<think>never closed"), ("never closed", ""))
        self.assertEqual(split_reasoning("plain"), ("", "plain"))

    def test_malformed_calls_are_reported_not_guessed(self) -> None:
        parsed = parse_generation("</think><tool_call>\n{not json}\n</tool_call>", "hermes")
        self.assertEqual(parsed.tool_calls, [])
        self.assertTrue(parsed.parse_errors)
        parsed = parse_generation("</think><tool_call>\n<function=x>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>", "qwen3_xml")
        self.assertEqual(parsed.tool_calls, [])

    def test_xml_scalars_stay_text(self) -> None:
        parsed = parse_generation(
            "</think>\n\n<tool_call>\n<function=lookup_name>\n<parameter=q>\n123\n</parameter>\n</function>\n</tool_call>",
            "qwen3_xml",
        )
        self.assertEqual(parsed.tool_calls[0]["arguments"], {"q": "123"})


class TemplateRoundTripTests(unittest.TestCase):
    def roundtrip(self, tokenizer_dir: str, tool_format: str) -> None:
        tokenizer = load_tokenizer(tokenizer_dir)
        rendered = tokenizer.apply_chat_template(to_template_messages(MESSAGES[:3]), tokenize=False)
        assistant = rendered.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        parsed = parse_generation(assistant, tool_format)
        self.assertEqual(parsed.parse_errors, [])
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0]["name"], CALL["name"])
        self.assertEqual(parsed.tool_calls[0]["arguments"], CALL["arguments"])
        self.assertEqual(parsed.reasoning, "Need a SMILES build.")

    def test_qwen35_xml_tool_calls_roundtrip(self) -> None:
        self.roundtrip("Qwen__Qwen3.5-9B", "qwen3_xml")

    def test_qwen3_hermes_tool_calls_roundtrip(self) -> None:
        self.roundtrip("Qwen__Qwen3-8B", "hermes")

    def test_tool_results_render_as_user_side(self) -> None:
        tokenizer = load_tokenizer("Qwen__Qwen3.5-9B")
        rendered = tokenizer.apply_chat_template(to_template_messages(MESSAGES), tokenize=False)
        self.assertIn("<tool_response>\n{\"status\":\"succeeded\"}\n</tool_response>", rendered)
        self.assertEqual(rendered.count("<|im_start|>assistant\n"), 2)


if __name__ == "__main__":
    unittest.main()
