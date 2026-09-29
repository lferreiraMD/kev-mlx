# kev-mlx

A web app for running [Kev](https://github.com/jaredpalmer/kev) — a small, Jev-style **decision model** —
locally on Apple Silicon. One process serves a browser UI and the TypeSafe-compatible API, backed by the
pre-quantized MLX build [`RoderickQiu/kev-4b-mlx-8bit`](https://huggingface.co/RoderickQiu/kev-4b-mlx-8bit):
the released kev-4b adapter merged into its Qwen3.5-4B base and quantized to 8 bits, with the released fp32
pointer head and calibration temperature.

Instead of generating text, Kev takes a **state** (any text or JSON) plus typed questions — yes/no (`noul`),
multiple choice (`choice`), rating (`score`) — and returns **calibrated probabilities** in a single prefill
pass. Warm requests take ~200 ms on an M-series Mac.

## Quick start

This repo is [jaredpalmer/kev](https://github.com/jaredpalmer/kev) (Apache-2.0; its original README is
preserved as [README_Jared.md](README_Jared.md), the deep dive on the models and training) plus the
`webapp/` folder, so one clone gets you everything:

```bash
git clone https://github.com/lferreiraMD/kev-mlx.git && cd kev-mlx
uv sync --extra serve
uv run --extra serve python webapp/server.py        # http://127.0.0.1:8009
```

The first run downloads the model from Hugging Face (~4.5 GB); after that it loads from the local cache and
works offline. Options: `--port`, `--host`, and `--model <hub-id>` for another merged+quantized MLX export.

## Using the app

Open [localhost:8009](http://127.0.0.1:8009):

- **State** — the content the questions are about. Plain text is sent as a string; paste JSON and it is sent
  structured (the badge next to the label shows which).
- **Questions** — add any mix of the three types. Each question sees the state and its own instructions and
  criteria, but never the other questions. For `choice`, options get a name and an optional description; for
  `score`, levels are ordered lowest to highest.
- **Run** (⌘⏎ / Ctrl+⏎) — answers appear under each question as probability bars with the picked option
  highlighted, plus confidence, model latency, and token usage. A collapsible panel shows the raw request and
  response JSON — useful as a template for calling the API from code.
- **Presets** — support ticket, content moderation, expense approval, and email triage show the shapes.

### The API

The same server exposes the TypeSafe-compatible endpoints from `kev.serve`:

```bash
curl -s localhost:8009/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice. Please fix this ASAP.",
  "model": "kev-latest",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this ticket about billing?"},
    "tone": {"type": "choice", "instructions": "What is the customer'\''s tone?",
             "criteria": {"calm": null, "frustrated": null, "angry": null}}
  }}'
```

The TypeSafe Python SDK works unchanged (installed by `uv sync --extra serve`):

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")
r = client.system_one(
    state="I was charged twice. Please fix this ASAP.",
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(instructions="What is the customer's tone?",
                       criteria={"calm": None, "frustrated": None, "angry": None}),
    },
)
print(r.nouls["billing"].noul, r.choices["tone"].choice)   # 0.85 frustrated
```

Also available: `GET /v1/models` (checkpoint details), `POST /v1/systemone/permute` (one choice question under
several option orders), `POST /v1/systemone/separate` (each question in its own forward pass). Set
`KEV_API_KEY` to require bearer auth; `--host 0.0.0.0` to serve beyond the machine.

## Why a separate launcher

The main repo's `Checkpoint` loader knows two layouts: a LoRA adapter on a base, or a full-weight torch
backbone. The 8-bit MLX export is a third — already merged and quantized — so `server.py` loads it directly
with mlx-lm through the repo's own `MLXDecisionModel`, skips the merge, loads the pointer head from the
export's `head.pt`, and hands the result to `kev.serve`'s `Server` and FastAPI app. Everything else
(encoding, prefix cache, batching, API) is the main repo's unmodified code.

## Decision models as an agent's decision layer

Most steps in a production agent are not writing — they are decisions: which tool to call, which model to
route to, whether a tool call is safe to execute, whether to escalate or stop. Using a System One model for
those decisions, instead of the main LLM, has become a recognized pattern since Jev's release, and it is a
good fit for a local Kev:

- **The decisions run out-of-band, so the agent's context stays clean.** The transcript never carries the
  deliberation, the re-ranked tool list, or the rejected options — the agent just receives the chosen tool.
  Kev answers each question independently and caches the state prefix, so asking several questions about the
  same context (which tool? escalate? did the last step fail?) costs roughly one request.
- **The model can only answer within the options you supply.** It cannot invent a tool that does not exist or
  return something off-schema, and the calibrated probabilities give you a principled threshold: act on
  confident decisions, **fail open** to the full LLM on the rest.
- **It is fast and cheap enough for the hot path** — ecosystem posts cite ~200× faster inference and ~400×
  lower cost than an LLM on classification-shaped decisions, and extra questions barely change latency.

What exists already:

- [AgentScope-Java PR #3240](https://github.com/agentscope-ai/agentscope-java/pull/3240) — System One
  middlewares for **tool selection** (a `choice` over the registered tools with a `__none__` threshold),
  **model routing**, and an **AutoMode safety gate** (a `noul` "is this call safe?" before guarded tools
  like bash). The review discussion is a catalog of the pitfalls: a `choice` softmax models *relative
  preference*, not independent applicability — if a step may need two tools, ask one `noul` per tool
  instead; chunking large tool sets loses recall; serial decision calls add hot-path latency; and fail-open
  needs to distinguish "the model said none" from "the model was unavailable".
- [LangChain: building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev) —
  recommends model routing and tool-risk gating; the decision model complements the LLM rather than
  replacing it.
- [REFLEX (arXiv:2609.26532)](https://arxiv.org/pdf/2609.26532) — delegates continue-vs-invoke-tool,
  escalation, and early-termination decisions to a System One model, calling the expensive LLM only where
  needed.
- [Calibrated decision models for pentest agents (arXiv:2609.28940)](https://arxiv.org/pdf/2609.28940) —
  Jev and Laya as decision layers in an LLM-driven pentest harness.

Caveats for this 4B build specifically: accuracy drops on long states (Kev-27B holds up much better),
and knowledge- or arithmetic-heavy decisions are weak — the expense-approval preset getting per-head math
wrong is a live demo of why the calibrated confidence matters. For agent workloads, the intended move is a
short fine-tune on your own traces: log the big model's actual decisions as labels and run the main repo's
[`kev-finetune`](https://github.com/jaredpalmer/kev/tree/main/skills/kev-finetune) skill (~$1 per Kev-4B
run on an H100); you get in-distribution accuracy plus a temperature fitted to your decisions, so the
confidence you threshold on means what it says.

## License

Apache-2.0. The upstream Kev code is unmodified and keeps its [LICENSE](LICENSE) and copyright
(original README: [README_Jared.md](README_Jared.md)); the `webapp/` additions are under the same license.
