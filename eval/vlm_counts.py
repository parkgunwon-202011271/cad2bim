"""Door/window counting on the same 30 held-out synthetic plans: our pipeline vs multimodal LLMs (AECV-Bench style).

사용: python eval/vlm_counts.py <합성 시험 폴더(synth2_bench.yaml 있는 곳)> [모델...]   models like openai:gpt-5.5 gemini:gemini-3.5-flash-lite
Metrics per class: exact-match accuracy of the count, mean absolute error, MAPE (AECV-Bench reports exact match + MAPE).
Our counts come from the rule pipeline's IR (bench_synth2), run without LLM.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
SYN = Path(sys.argv[1])
MODELS = sys.argv[2:] or ["openai:gpt-5.5", "gemini:gemini-3.5-flash-lite"]
OUT = Path(__file__).resolve().parent / "results" / "vlm"
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

from cad2bim.envfile import load_env  # noqa: E402

load_env()
from cad2bim import llm_client  # noqa: E402
from cad2bim.llm_client import ask, configure, extract_json  # noqa: E402

PROMPT = ("This image is an architectural floor plan (a CAD drawing). Count every door and every window drawn in it. "
          "Doors include swing, double, sliding, shutter and elevator doors. Windows include glazing and curtain-wall panels "
          "drawn as windows. Reply with JSON only: {\"doors\": <integer>, \"windows\": <integer>}")


def render(dxf: Path, png: Path):
    if png.exists():
        return
    import ezdxf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(dxf)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_axes([0, 0, 1, 1])
    cfg = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(doc.modelspace(), finalize=True)
    fig.savefig(png, dpi=100)
    plt.close(fig)


manifest = yaml.safe_load(open(SYN / "synth2_bench.yaml", encoding="utf-8"))
items = manifest.get("drawings") or manifest.get("items") or next(v for v in manifest.values() if isinstance(v, list))
rows = []
for it in items:
    dxf = SYN / it["dxf"]
    truth = json.load(open(SYN / it["truth"], encoding="utf-8"))
    t = {"doors": sum(o["label"] == "door" for o in truth["openings"]), "windows": sum(o["label"] == "window" for o in truth["openings"])}
    ours = {"doors": None, "windows": None}  # 우리 개수는 eval/score_counts.py(조립 결과)로 잰다
    png = IMG / f"{it['id']}.png"
    render(dxf, png)
    rows.append({"id": it["id"], "truth": t, "ours": ours, "png": str(png)})
print("plans", len(rows), "rendered", flush=True)

cache_path = Path(os.environ.get("VLM_CACHE", str(OUT / "answers.json")))
cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
for spec in MODELS:
    provider, model = spec.split(":", 1)
    rpm = 0 if provider != "gemini" else (14 if "lite" in model else 4)
    configure({"llm": {"requests_per_minute": rpm}})
    limit = int(os.environ.get("VLM_LIMIT", "0")) or len(rows)  # free tier: stronger models allow 20 calls a day
    for r in rows[:limit]:
        key = f"{spec}|{r['id']}"
        if key in cache:
            continue
        t0 = time.time()
        try:
            text = ask(provider, model, PROMPT, Path(r["png"]).read_bytes(), max_tokens=32000)
            ans = extract_json(text) or {"raw": (text or "")[-300:]}
        except Exception as e:  # keep going; record the failure
            ans = {"error": str(e)[:120]}
        ans["seconds"] = round(time.time() - t0, 1)
        cache[key] = ans
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(spec, r["id"], ans, flush=True)


def score(get):
    res = {}
    for cls in ("doors", "windows"):
        exact = mae = ape = n = 0
        for r in rows:
            v = get(r)
            if v is None or not isinstance(v.get(cls), (int, float)):
                v = {cls: 0}
            p, t = int(v[cls]), r["truth"][cls]
            exact += p == t
            mae += abs(p - t)
            ape += abs(p - t) / max(t, 1)
            n += 1
        res[cls] = {"exact": round(exact / n, 3), "mae": round(mae / n, 2), "mape": round(100 * ape / n, 1)}
    return res


def ok(spec, r):
    v = cache.get(f"{spec}|{r['id']}")
    return isinstance(v, dict) and isinstance(v.get("doors"), (int, float)) and isinstance(v.get("windows"), (int, float))


all_rows = rows
rows = [r for r in all_rows if all(ok(spec, r) for spec in MODELS)]  # score every side on the same plans only
print("plans answered by every model:", len(rows), "of", len(all_rows))
table = {}
for spec in MODELS:
    table[spec] = score(lambda r, s=spec: cache.get(f"{s}|{r['id']}"))
json.dump({"plans": [r["id"] for r in rows], "rows": all_rows, "table": table}, open(OUT / "result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in table.items():
    print(f"{k:32s} doors exact {v['doors']['exact']:.2f} MAE {v['doors']['mae']:5.1f} MAPE {v['doors']['mape']:5.1f}% | "
          f"windows exact {v['windows']['exact']:.2f} MAE {v['windows']['mae']:5.1f} MAPE {v['windows']['mape']:5.1f}%")
