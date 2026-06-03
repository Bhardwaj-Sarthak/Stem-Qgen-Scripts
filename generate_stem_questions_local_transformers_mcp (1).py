#!/usr/bin/env python3
"""
Generate STEM question-answer pairs with a LOCAL Hugging Face Transformers model
while using the STEM-QA-Contextualize MCP endpoint for retrieval and/or scoring.

Pipelines:
  default   : local model generates directly
  rag       : MCP QuestionRetrieverTool -> local model generates with examples
  tool      : local model generates -> MCP classify_levels_phrases -> revise locally
  rag_tool  : MCP QuestionRetrieverTool -> local model generates -> MCP classify -> revise

Full thesis-style sweep:
  3 subjects x 4 grades/classes x 2 topics x 6 Bloom levels x 4 DOK levels
  = 576 rows per pipeline.

Install:
  pip install "mcp>=1.9" "transformers>=4.45" accelerate torch tqdm pandas
  # optional for 4-bit/8-bit quantization:
  pip install bitsandbytes

Examples:
  python generate_stem_questions_local_transformers_mcp.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --pipelines rag_tool \
    --limit-combos 10 \
    --out test_local_questions.jsonl \
    --csv-out test_local_questions.csv

  python generate_stem_questions_local_transformers_mcp.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --hf-token "$HF_TOKEN" \
    --load-in-4bit \
    --pipelines default rag tool rag_tool
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from stem_prompt_templates import build_pipeline_prompt
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install transformers accelerate torch") from exc

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError as exc:
    raise SystemExit('Missing dependency: pip install "mcp>=1.9"') from exc

try:
    from mcp.client.sse import sse_client
except Exception:  # pragma: no cover
    sse_client = None


DEFAULT_MCP_URL = "https://bhardwaj08sarthak-stem-qa-contextualize-mcp.hf.space/gradio_api/mcp/"

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
DOK_LEVELS = ["DOK1", "DOK2", "DOK3", "DOK4"]

CURRICULUM: Dict[str, Dict[str, List[str]]] = {
    "Maths": {
        "Grade 5": ["Fractions", "Decimals"],
        "Grade 8": ["Linear equations", "Geometry: angles"],
        "Grade 10": ["Quadratic equations", "Probability"],
        "Grade 12": ["Calculus: derivatives", "Vectors"],
    },
    "Physics": {
        "Grade 5": ["Forces and motion (basic)", "Energy (forms)"],
        "Grade 8": ["Newton's laws", "Work, power, energy"],
        "Grade 10": ["Electricity and circuits", "Projectile motion"],
        "Grade 12": ["Electromagnetic induction", "Thermodynamics"],
    },
    "Chemistry": {
        "Grade 5": ["States of matter", "Mixtures and solutions"],
        "Grade 8": ["Atoms and molecules", "Chemical reactions (basics)"],
        "Grade 10": ["Mole concept (intro)", "Periodic trends"],
        "Grade 12": ["Chemical equilibrium", "Organic chemistry: mechanisms"],
    },
}

PIPELINES = ("default", "rag", "tool", "rag_tool")


@dataclass(frozen=True)
class GenerationSpec:
    subject: str
    grade: str
    topic: str
    target_bloom: str
    target_dok: str
    pipeline: str
    index: int


@dataclass
class QARecord:
    spec: Dict[str, Any]
    model_id: str
    question: str
    answer: str
    reasoning: str
    examples: List[str]
    classification: Optional[Any]
    attempts: int
    ok: bool
    error: Optional[str] = None


def iter_specs(pipelines: Sequence[str]) -> Iterable[GenerationSpec]:
    i = 0
    for pipeline in pipelines:
        for subject, by_grade in CURRICULUM.items():
            for grade, topics in by_grade.items():
                for topic, bloom, dok in product(topics, BLOOM_LEVELS, DOK_LEVELS):
                    i += 1
                    yield GenerationSpec(subject, grade, topic, bloom, dok, pipeline, i)


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def compact_tool_result(result: Any) -> Any:
    """Convert an MCP SDK CallToolResult into plain Python objects/text."""
    if hasattr(result, "content"):
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        joined = "\n".join(parts).strip()
        try:
            return json.loads(joined)
        except Exception:
            return joined
    return result


def parse_retrieved_examples(raw: Any) -> List[str]:
    raw = compact_tool_result(raw)
    if isinstance(raw, str):
        try:
            return parse_retrieved_examples(json.loads(raw))
        except Exception:
            return [raw] if raw else []
    if isinstance(raw, dict):
        for key, val in raw.items():
            if "questions" in str(key).lower() and isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, dict) and "text" in item:
                        out.append(str(item["text"]))
                    else:
                        out.append(str(item))
                return out[:5]
    return []


def get_best_levels(classification: Any) -> Tuple[Optional[str], Optional[str]]:
    c = compact_tool_result(classification)
    if isinstance(c, str):
        # Gradio may serialize a Python-ish dict string. This regex finds both Bloom and DOK best labels.
        matches = re.findall(r"the question is best aligned to -['\"]?\s*:\s*['\"]([^'\"]+)", c)
        if matches:
            return matches[0], matches[-1]
        return None, None
    if isinstance(c, dict):
        bloom_best = None
        dok_best = None
        for k, v in c.items():
            lk = str(k).lower()
            if isinstance(v, dict) and "bloom" in lk:
                bloom_best = v.get("the question is best aligned to -")
            if isinstance(v, dict) and "dok" in lk:
                dok_best = v.get("the question is best aligned to -")
        return bloom_best, dok_best
    return None, None


def levels_match(pred_bloom: Optional[str], pred_dok: Optional[str], target_bloom: str, target_dok: str) -> bool:
    def norm(x: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())

    return norm(pred_bloom) == norm(target_bloom) and norm(pred_dok) == norm(target_dok)


class StemMCPClient:
    def __init__(self, url: str, hf_token: Optional[str] = None) -> None:
        self.url = url.rstrip("/") + "/"
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
        self.session: Optional[ClientSession] = None
        self._cm = None
        self.tool_names: List[str] = []

    async def __aenter__(self) -> "StemMCPClient":
        try:
            self._cm = streamablehttp_client(self.url, headers=self.headers)
            read, write, _ = await self._cm.__aenter__()
        except Exception:
            if sse_client is None:
                raise
            sse_url = self.url.rstrip("/") + "/sse"
            self._cm = sse_client(sse_url, headers=self.headers)
            read, write = await self._cm.__aenter__()
        self.session = ClientSession(read, write)
        await self.session.__aenter__()
        await self.session.initialize()
        tools = await self.session.list_tools()
        self.tool_names = [t.name for t in tools.tools]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self.session is not None:
            await self.session.__aexit__(exc_type, exc, tb)
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc, tb)

    def _tool(self, *candidates: str) -> str:
        lower = {name.lower(): name for name in self.tool_names}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        for name in self.tool_names:
            lname = name.lower()
            if any(c.lower() in lname for c in candidates):
                return name
        raise RuntimeError(f"Could not find tool among {self.tool_names}; tried {candidates}")

    async def retrieve(self, subject: str, topic: str, grade: str) -> List[str]:
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        tool = self._tool("QuestionRetrieverTool", "questionretrievertool", "retrieve")
        res = await self.session.call_tool(tool, {"subject": subject, "topic": topic, "grade": grade})
        return parse_retrieved_examples(res)

    async def classify(self, question: str) -> Any:
        if self.session is None:
            raise RuntimeError("MCP session not initialized")
        tool = self._tool("classify_levels_phrases", "classify", "score")
        return compact_tool_result(await self.session.call_tool(tool, {"question": question}))


class LocalTransformersGenerator:
    def __init__(
        self,
        model_id: str,
        hf_token: Optional[str],
        torch_dtype: str,
        device_map: str,
        load_in_4bit: bool,
        load_in_8bit: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        dtype = self._resolve_dtype(torch_dtype)
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise SystemExit("For --load-in-4bit/--load-in-8bit install bitsandbytes") from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
            )

        print(f"Loading local model: {model_id}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
        )
        self.model.eval()

    @staticmethod
    def _resolve_dtype(torch_dtype: str):
        value = torch_dtype.lower()
        if value == "auto":
            return "auto"
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        if value in {"fp32", "float32"}:
            return torch.float32
        raise ValueError("--torch-dtype must be auto, bf16, fp16, or fp32")

    def _build_prompt(
    self,
    spec: GenerationSpec,
    examples: Sequence[str] = (),
    feedback: Optional[str] = None,
    previous_question: Optional[str] = None,) -> str:
        return build_pipeline_prompt(
            spec=spec,
            examples=examples,
            feedback=feedback,
            previous_question=previous_question,
        )

    def _generate_once(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt")
        # Move tensors to the first model device. With device_map='auto', accelerate handles dispatch.
        try:
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        except Exception:
            pass

        do_sample = self.temperature > 0
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    async def generate(
        self,
        spec: GenerationSpec,
        examples: Sequence[str] = (),
        feedback: Optional[str] = None,
        previous_question: Optional[str] = None,
    ) -> Dict[str, str]:
        prompt = self._build_prompt(spec, examples, feedback, previous_question)
        # Run blocking local generation in a thread so MCP I/O remains async-safe.
        raw = await asyncio.to_thread(self._generate_once, prompt)
        obj = extract_json_object(raw)
        return {
            "question": str(obj.get("question", "")).strip(),
            "answer": str(obj.get("answer", "")).strip(),
            "reasoning": str(obj.get("reasoning", "")).strip(),
        }


async def run_one(spec: GenerationSpec, gen: LocalTransformersGenerator, mcp: StemMCPClient, attempts: int) -> QARecord:
    examples: List[str] = []
    classification: Optional[Any] = None
    feedback: Optional[str] = None
    previous_question: Optional[str] = None

    try:
        if spec.pipeline in {"rag", "rag_tool"}:
            examples = await mcp.retrieve(spec.subject, spec.topic, spec.grade)

        max_attempts = attempts if spec.pipeline in {"tool", "rag_tool"} else 1
        qa = {"question": "", "answer": "", "reasoning": ""}
        ok = False

        for attempt in range(1, max_attempts + 1):
            qa = await gen.generate(spec, examples=examples, feedback=feedback, previous_question=previous_question)
            if spec.pipeline not in {"tool", "rag_tool"}:
                return QARecord(asdict(spec), gen.model_id, qa["question"], qa["answer"], qa["reasoning"], examples, None, attempt, True)

            classification = await mcp.classify(qa["question"])
            pred_bloom, pred_dok = get_best_levels(classification)
            ok = levels_match(pred_bloom, pred_dok, spec.target_bloom, spec.target_dok)
            if ok:
                return QARecord(asdict(spec), gen.model_id, qa["question"], qa["answer"], qa["reasoning"], examples, classification, attempt, True)

            previous_question = qa["question"]
            feedback = json.dumps(
                {
                    "target_bloom": spec.target_bloom,
                    "target_dok": spec.target_dok,
                    "predicted_bloom": pred_bloom,
                    "predicted_dok": pred_dok,
                    "raw_classifier": classification,
                },
                ensure_ascii=False,
            )[:4000]

        return QARecord(asdict(spec), gen.model_id, qa["question"], qa["answer"], qa["reasoning"], examples, classification, max_attempts, ok)
    except Exception as exc:
        return QARecord(asdict(spec), gen.model_id, "", "", "", examples, classification, 0, False, error=repr(exc))


def write_jsonl(path: Path, rows: Sequence[QARecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[QARecord]) -> None:
    fields = [
        "model_id",
        "pipeline",
        "subject",
        "grade",
        "topic",
        "target_bloom",
        "target_dok",
        "question",
        "answer",
        "reasoning",
        "attempts",
        "ok",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            s = r.spec
            writer.writerow(
                {
                    "model_id": r.model_id,
                    "pipeline": s["pipeline"],
                    "subject": s["subject"],
                    "grade": s["grade"],
                    "topic": s["topic"],
                    "target_bloom": s["target_bloom"],
                    "target_dok": s["target_dok"],
                    "question": r.question,
                    "answer": r.answer,
                    "reasoning": r.reasoning,
                    "attempts": r.attempts,
                    "ok": r.ok,
                    "error": r.error or "",
                }
            )


async def main_async(args: argparse.Namespace) -> int:
    pipelines = [p.strip() for p in args.pipelines]
    bad = sorted(set(pipelines) - set(PIPELINES))
    if bad:
        raise SystemExit(f"Unknown pipeline(s): {bad}. Choose from {PIPELINES}")

    specs = list(iter_specs(pipelines))
    if args.limit_combos:
        specs = specs[: args.limit_combos]

    gen = LocalTransformersGenerator(
        model_id=args.model_id,
        hf_token=args.hf_token,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    rows: List[QARecord] = []
    async with StemMCPClient(args.mcp_url, hf_token=args.hf_token) as mcp:
        print(f"Connected to MCP tools: {mcp.tool_names}", file=sys.stderr)
        for spec in tqdm(specs, desc="Generating"):
            rows.append(await run_one(spec, gen, mcp, attempts=args.attempts))
            if args.sleep > 0:
                await asyncio.sleep(args.sleep)

    out = Path(args.out)
    write_jsonl(out, rows)
    if args.csv_out:
        write_csv(Path(args.csv_out), rows)
    print(f"Wrote {len(rows)} records to {out}")
    if args.csv_out:
        print(f"Wrote CSV to {args.csv_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate STEM QA pairs with local Hugging Face Transformers + MCP.")
    parser.add_argument("--mcp-url", default=os.getenv("MCP_URL", DEFAULT_MCP_URL), help="Remote Gradio MCP URL")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"), help="HF token for gated models/private Spaces, optional")
    parser.add_argument("--model-id", default=os.getenv("HF_LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct"), help="HF model id or local model path")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map, e.g. auto, cuda:0, cpu")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit quantization")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use bitsandbytes 8-bit quantization")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--pipelines", nargs="+", default=["rag_tool"], choices=PIPELINES)
    parser.add_argument("--attempts", type=int, default=3, help="Revision attempts for tool/rag_tool pipelines")
    parser.add_argument("--limit-combos", type=int, default=0, help="For quick tests; 0 means full sweep")
    parser.add_argument("--out", default="stem_questions_local.jsonl")
    parser.add_argument("--csv-out", default="stem_questions_local.csv")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between generations to reduce MCP/Space load")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit or --load-in-8bit")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
