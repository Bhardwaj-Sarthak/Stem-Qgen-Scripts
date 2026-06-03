"""
Pipeline-specific prompt templates for STEM question generation.

Import these templates in generate_stem_questions_local_transformers_mcp.py
and call build_pipeline_prompt(...) from LocalTransformersGenerator._build_prompt(...).

The four pipelines are:
  default   : local model generates directly
  rag       : MCP QuestionRetrieverTool -> local model generates with examples
  tool      : local model generates -> MCP classify_levels_phrases -> revise locally
  rag_tool  : MCP QuestionRetrieverTool -> local model generates -> MCP classify -> revise
"""

from __future__ import annotations

from typing import Optional, Sequence, Protocol


class GenerationSpecLike(Protocol):
    """Minimal interface expected from the main script's GenerationSpec dataclass."""

    subject: str
    grade: str
    topic: str
    target_bloom: str
    target_dok: str
    pipeline: str


DEFAULT_PROMPT = """You are an expert STEM educational item writer and psychometric reviewer.

Generate exactly one grade-appropriate STEM question-answer pair.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Bloom's Taxonomy target: {target_bloom}
- Depth of Knowledge target: {target_dok}

Cognitive-demand rules:
- Bloom's level controls the learning objective of the question.
- DOK level controls the depth of reasoning required to answer it.
- Do not merely insert a Bloom/DOK keyword; make the actual reasoning demand match the target.
- The item must be technically correct, unambiguous, and appropriate for the specified grade.

Question-design constraints:
- Ask only one main question.
- Avoid overly generic textbook wording.
- Avoid giving away the answer in the question.
- Use numbers, scenarios, diagrams-in-words, or short contexts when useful.
- Keep the question answerable without external resources.
- For maths/physics, include units where relevant.
- For chemistry, ensure terminology is age-appropriate.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should briefly explain why the item fits the target Bloom and DOK levels.
"""


RAG_PROMPT = """You are an expert STEM educational item writer.

You are given retrieved benchmark-style examples from an MCP retrieval tool.
Use them only as inspiration for style, domain vocabulary, grade level, and item structure.
Do NOT copy, paraphrase, or lightly modify any retrieved example.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Bloom's Taxonomy target: {target_bloom}
- Depth of Knowledge target: {target_dok}

Retrieved examples:
{examples_block}

Your task:
Generate exactly one new question-answer pair that is clearly about the target topic and suitable for the target grade.

RAG-use rules:
- Match the educational style and difficulty of the retrieved examples.
- Preserve the target topic, not just the broad subject.
- Use the retrieved examples to calibrate vocabulary and structure.
- Do not reuse names, values, wording, or final answers from the examples.
- If the examples are only loosely related, prioritize the target topic and grade over the examples.

Cognitive-demand rules:
- Bloom's level should determine the learning objective.
- DOK level should determine how much reasoning, planning, or explanation is required.
- Make the cognitive demand real, not just keyword-based.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should mention how the question uses the retrieved examples without copying them, and why it fits the Bloom/DOK target.
"""


TOOL_FIRST_ATTEMPT_PROMPT = """You are an expert STEM educational item writer and cognitive-demand optimizer.

Generate exactly one STEM question-answer pair.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Bloom's Taxonomy target: {target_bloom}
- Depth of Knowledge target: {target_dok}

Important:
This question will be checked by an external MCP classifier for Bloom and DOK alignment.
Write the question so that the classifier and a human reviewer would both identify it as matching the requested levels.

Cognitive-demand rules:
- Bloom's target: shape the learning objective.
- DOK target: shape the reasoning depth.
- Do not rely only on action verbs.
- Make the solution process actually require the target level of thinking.
- For DOK1, use direct recall or one-step work.
- For DOK2, require classification, comparison, explanation, or routine multi-step reasoning.
- For DOK3, require justification, strategy, evidence, or non-routine reasoning.
- For DOK4, require synthesis, design, modeling, critique, or extended reasoning.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should explicitly justify the Bloom and DOK alignment.
"""


TOOL_REVISION_PROMPT = """You are revising a STEM question after external MCP cognitive classification.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Target Bloom's Taxonomy level: {target_bloom}
- Target Depth of Knowledge level: {target_dok}

Previous question:
{previous_question}

MCP classifier feedback:
{feedback}

Revision task:
Generate a revised question-answer pair that better matches BOTH target levels.

Revision rules:
- Keep the same subject, grade, and topic.
- Change the actual task demand, not just the wording.
- If Bloom was too low, require a higher-order learning objective.
- If Bloom was too high, simplify the objective.
- If DOK was too low, add reasoning, evidence, planning, justification, or non-routine decision-making.
- If DOK was too high, reduce the number of steps and make the task more direct.
- Do not mention the classifier in the final JSON.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should explain why the revised version now better fits the target Bloom and DOK levels.
"""


RAG_TOOL_FIRST_ATTEMPT_PROMPT = """You are an expert STEM educational item writer using retrieved benchmark examples and cognitive-alignment checking.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Bloom's Taxonomy target: {target_bloom}
- Depth of Knowledge target: {target_dok}

Retrieved benchmark-style examples from MCP:
{examples_block}

Your task:
Generate exactly one NEW question-answer pair that is:
1. aligned with the retrieved examples in style and domain vocabulary,
2. appropriate for the specified grade,
3. clearly about the target topic,
4. aligned with the requested Bloom and DOK levels.

Rules for using examples:
- Use examples as calibration, not source text.
- Do not copy or paraphrase retrieved questions.
- Do not reuse numerical values, names, scenarios, or final answers from examples.
- If examples conflict with the target grade/topic, follow the target grade/topic.

Cognitive-demand rules:
- Bloom target defines the learning objective.
- DOK target defines the reasoning depth.
- The question should be classifiable as the requested Bloom/DOK level by both keyword cues and actual solution process.
- Avoid superficial action-verb matching.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should briefly explain:
- how the retrieved examples influenced the style,
- why the item fits the target Bloom level,
- why the item fits the target DOK level.
"""


RAG_TOOL_REVISION_PROMPT = """You are revising a retrieved-context STEM question after external MCP cognitive classification.

Generation target:
- Subject: {subject}
- Grade/Class: {grade}
- Topic: {topic}
- Target Bloom's Taxonomy level: {target_bloom}
- Target Depth of Knowledge level: {target_dok}

Retrieved benchmark-style examples:
{examples_block}

Previous generated question:
{previous_question}

MCP classifier feedback:
{feedback}

Revision task:
Generate a revised question-answer pair that improves cognitive alignment while preserving benchmark-style domain grounding.

Revision rules:
- Keep the item within the same subject, grade, and topic.
- Preserve the style and domain vocabulary suggested by the retrieved examples.
- Do not copy the examples or the previous question.
- Modify the actual reasoning structure of the question.
- If the classifier predicted the wrong Bloom level, adjust the learning objective.
- If the classifier predicted the wrong DOK level, adjust the depth of reasoning required.
- The final item should be technically correct and solvable.

Return ONLY valid compact JSON with exactly these keys:
{{"question":"...","answer":"...","reasoning":"..."}}

The reasoning field should explain how the revision changed the cognitive demand.
"""


PROMPTS_BY_PIPELINE = {
    "default": DEFAULT_PROMPT,
    "rag": RAG_PROMPT,
    "tool_first": TOOL_FIRST_ATTEMPT_PROMPT,
    "tool_revision": TOOL_REVISION_PROMPT,
    "rag_tool_first": RAG_TOOL_FIRST_ATTEMPT_PROMPT,
    "rag_tool_revision": RAG_TOOL_REVISION_PROMPT,
}


def format_examples_block(examples: Sequence[str], max_examples: int = 5, max_chars_per_example: int = 1200) -> str:
    """Format retrieved MCP examples for insertion into RAG prompts."""
    if not examples:
        return "No examples retrieved."

    formatted = []
    for idx, example in enumerate(examples[:max_examples], start=1):
        clean = str(example).strip().replace("\n", " ")
        if len(clean) > max_chars_per_example:
            clean = clean[:max_chars_per_example].rstrip() + "..."
        formatted.append(f"{idx}. {clean}")
    return "\n".join(formatted)


def build_pipeline_prompt(
    spec: GenerationSpecLike,
    examples: Sequence[str] = (),
    feedback: Optional[str] = None,
    previous_question: Optional[str] = None,
) -> str:
    """
    Build the appropriate prompt for one pipeline/spec.

    Parameters
    ----------
    spec:
        The main script's GenerationSpec instance, or any object with the same fields.
    examples:
        Retrieved examples from the MCP QuestionRetrieverTool. Used by rag/rag_tool prompts.
    feedback:
        Serialized classifier feedback from the MCP classifier. Used by tool/rag_tool revision prompts.
    previous_question:
        The previous generated question. Used by revision prompts.
    """
    values = {
        "subject": spec.subject,
        "grade": spec.grade,
        "topic": spec.topic,
        "target_bloom": spec.target_bloom,
        "target_dok": spec.target_dok,
        "examples_block": format_examples_block(examples),
        "feedback": feedback or "No classifier feedback provided.",
        "previous_question": previous_question or "No previous question provided.",
    }

    if spec.pipeline == "default":
        template = DEFAULT_PROMPT
    elif spec.pipeline == "rag":
        template = RAG_PROMPT
    elif spec.pipeline == "tool":
        template = TOOL_REVISION_PROMPT if feedback else TOOL_FIRST_ATTEMPT_PROMPT
    elif spec.pipeline == "rag_tool":
        template = RAG_TOOL_REVISION_PROMPT if feedback else RAG_TOOL_FIRST_ATTEMPT_PROMPT
    else:
        raise ValueError(f"Unknown pipeline: {spec.pipeline}")

    return template.format(**values)
