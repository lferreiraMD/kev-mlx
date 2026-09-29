"""Kev web app: serve the pre-quantized MLX build (RoderickQiu/kev-4b-mlx-8bit) with the kev API plus a browser UI.

The repo's Checkpoint loader knows two layouts: a LoRA adapter on a base, or a full-weight torch backbone. This
export is a third one - the kev-4b adapter already merged into the bf16 base and quantized with mlx.nn.quantize
(bits=8, group_size=64), plus the released fp32 pointer head in head.pt (see its provenance.json). mlx-lm loads
such a directory directly, so this launcher feeds it into the repo's own MLXDecisionModel, skips merge_lora,
loads the head from head.pt itself, and hands the result to kev.serve's Server and FastAPI app. The tokenizer
comes from the export too (it carries the base's tokenizer files, delimiters included), so nothing else downloads.

Run:  uv run --extra serve python webapp/server.py            # http://127.0.0.1:8009
Then open the printed URL; the UI lives at /, the TypeSafe-compatible API at /v1/systemone as usual.
"""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from kev.checkpoint import Checkpoint
from kev.mlx_model import MLXDecisionModel
from kev.model import PointerHead, pad_id

DEFAULT_MODEL = "RoderickQiu/kev-4b-mlx-8bit"
SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]   # what resolve_run pulls, too


class QuantizedKev(MLXDecisionModel):
    """MLXDecisionModel over a pre-merged, pre-quantized export: nothing to merge, head loaded from the export."""

    def __init__(self, model_dir, tok, meta):
        super().__init__(model_dir, pad_id(tok), head_dim=meta.head_dim)
        # the parent sized the head from embed_tokens.weight, which quantization packs 4-to-1; rebuild it from head.pt
        self.head = PointerHead(meta.head["q.weight"].shape[1], dp=meta.head_dim).eval()
        self.head.load_state_dict(meta.head)
        self.head.temperature = meta.temperature
        import json
        self.quantization = json.loads((Path(model_dir) / "config.json").read_text()).get("quantization") or {}

    @property
    def dtype(self):
        q = self.quantization
        return f"q{q.get('bits', '?')} (group {q.get('group_size', '?')})" if q else super().dtype


def load_quantized(repo):
    """-> (Checkpoint, tokenizer, model) for a merged+quantized MLX export, from the local HF cache first."""
    from huggingface_hub import snapshot_download
    try:
        snap = snapshot_download(repo, local_files_only=True, allow_patterns=SNAPSHOT_PATTERNS)
    except Exception:
        snap = snapshot_download(repo, allow_patterns=SNAPSHOT_PATTERNS)
    ck = Checkpoint(snap)
    ck.requested = repo   # labels and /v1/models report the Hub id, not the cache path
    tok = AutoTokenizer.from_pretrained(snap)
    model = QuantizedKev(snap, tok, ck.meta)
    return ck, tok, model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hub id of a merged+quantized MLX Kev export")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    a = ap.parse_args()

    print(f"loading {a.model} ...")
    ck, tok, model = load_quantized(a.model)

    from fastapi.staticfiles import StaticFiles
    import kev.serve as serve
    serve.app.state.server = serve.Server(ck, tok, model, "mlx")
    serve.app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

    print(f"serving {ck.requested} on mlx ({model.dtype}), temperature {model.head.temperature:.2f}")
    print(f"UI: http://{a.host}:{a.port}  |  API: POST http://{a.host}:{a.port}/v1/systemone")
    import uvicorn
    uvicorn.run(serve.app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
