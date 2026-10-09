"""Local notebook collection served through marimo's public ASGI API."""

import argparse
import json
from pathlib import Path

import marimo
import uvicorn
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = json.loads(Path(__file__).with_name("catalog.json").read_text())
DOCUMENTS = {
    "guide": ("Study guide", "README.md"),
    "roadmap": ("API status", "ROADMAP.md"),
    "validation": ("Validation", "VALIDATION.md"),
}


def document_html(title, filename):
    source = (ROOT / filename).read_text()
    for notebook in NOTEBOOKS:
        source = source.replace(
            f"(notebooks/{notebook['file']})", f"(/notebooks/{notebook['id']}/)"
        )
    for slug, (_, document) in DOCUMENTS.items():
        source = source.replace(f"({document})", f"(/{slug})")
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title} · Metatomic Lab</title>"
        "<style>body{font:16px/1.7 system-ui;color:#1d332c;background:#fafbf7;"
        "max-width:1000px;margin:40px auto;padding:0 24px}"
        "h1,h2,h3{line-height:1.25;margin-top:2em}h1{font-family:Georgia,serif}"
        "a{color:#286349}nav{display:flex;gap:24px;flex-wrap:wrap}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:14px}"
        "td,th{padding:12px;border:1px solid #dce3da;text-align:left;vertical-align:top}"
        "pre{background:#edf2e8;padding:18px;overflow:auto;border-radius:6px}"
        "code{font-size:.9em}p{max-width:80ch}</style>"
        "<nav><a href='/'>Notebook Lab</a><a href='/guide'>Guide</a>"
        "<a href='/roadmap'>API status</a><a href='/validation'>Validation</a></nav>"
        + marimo.md(source).text
        + "</html>"
    )


def create_app():
    builder = marimo.create_asgi_app(include_code=True, quiet=True, session_ttl=120)
    for notebook in NOTEBOOKS:
        path = ROOT / "notebooks" / notebook["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"Notebook not found: {path}. Run from the metatomic-marimo checkout."
            )
        builder = builder.with_app(path=f"/notebooks/{notebook['id']}", root=str(path))

    async def home(request):
        return HTMLResponse(Path(__file__).with_name("lab.html").read_text())

    async def catalog(request):
        return JSONResponse(NOTEBOOKS)

    async def source(request):
        notebook = next(
            (n for n in NOTEBOOKS if n["id"] == request.path_params["slug"]), None
        )
        if notebook is None:
            return PlainTextResponse("Notebook not found", status_code=404)
        return PlainTextResponse((ROOT / "notebooks" / notebook["file"]).read_text())

    async def document(request):
        title, filename = DOCUMENTS[request.url.path.strip("/")]
        return HTMLResponse(document_html(title, filename))

    return Starlette(
        routes=[
            Route("/", home),
            Route("/api/notebooks", catalog),
            *(Route(f"/{slug}", document) for slug in DOCUMENTS),
            Route("/source/{slug}", source),
            Mount("/", app=builder.build()),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start the local metatomic notebook lab"
    )
    parser.add_argument("--port", type=int, default=2719)
    args = parser.parse_args()
    print(f"Metatomic notebook lab: http://localhost:{args.port}", flush=True)
    # ASGI kernels share a process. Initialize native registrations and plotting
    # once before concurrent sessions import these modules from worker threads.
    import importlib
    import matplotlib

    # The ASGI kernels render figures in worker threads; native macOS GUI
    # backends require the main thread. Agg supplies the inline image instead.
    matplotlib.use("Agg")
    for backend in ("numpy", "torch", "jax", "matplotlib.pyplot"):
        importlib.import_module(backend)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
