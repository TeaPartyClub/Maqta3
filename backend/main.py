"""
Maqta3 — Arabic Reels Generator · FastAPI Server
-------------------------------------------------
Run:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000
"""

import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence chatty third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ─── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent          # …/Maqta3/
BACKEND_DIR  = Path(__file__).parent                 # …/Maqta3/backend/
JOBS_DIR     = BACKEND_DIR / "jobs"
UPLOADS_DIR  = BACKEND_DIR / "uploads"
RESULTS_DIR  = BACKEND_DIR / "results"

for _d in (JOBS_DIR, UPLOADS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Maqta3", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT_DIR), name="static")

# Lazy-import pipeline so the server starts fast (models load on first job)
_processor = None

def get_processor():
    global _processor
    if _processor is None:
        from pipeline import VideoProcessor
        _processor = VideoProcessor()
    return _processor


# ─── Job persistence ───────────────────────────────────────────────────────────
def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"

def read_job(job_id: str) -> dict:
    p = job_path(job_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(p.read_text(encoding="utf-8"))

def write_job(job_id: str, data: dict):
    job_path(job_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def patch_job(job_id: str, **kwargs):
    data = read_job(job_id)
    data.update(kwargs)
    write_job(job_id, data)


# ─── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html = ROOT_DIR / "Maqta3.html"
    return html.read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    url:            Optional[str]        = Form(None),
    file:           Optional[UploadFile] = File(None),
    num_highlights: int                  = Form(5),
):
    if not url and not file:
        raise HTTPException(400, "Provide a YouTube URL or upload a video file.")

    job_id   = str(uuid.uuid4())
    job_dir  = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Persist uploaded file
    upload_path: Optional[str] = None
    if file:
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        upload_path = str(UPLOADS_DIR / f"{job_id}{suffix}")
        async with aiofiles.open(upload_path, "wb") as fout:
            await fout.write(await file.read())

    write_job(job_id, {
        "job_id":   job_id,
        "status":   "pending",
        "stage":    None,
        "progress": 0,
        "message":  "Queued",
        "results":  [],
        "error":    None,
    })

    background_tasks.add_task(
        _run_pipeline, job_id, url, upload_path, num_highlights
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    return read_job(job_id)


@app.get("/api/results/{job_id}/{filename}")
async def get_result_file(job_id: str, filename: str):
    # Sanitise — allow only simple filenames (no path traversal)
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename")
    path = RESULTS_DIR / job_id / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=filename)


# ─── Background pipeline runner ────────────────────────────────────────────────
def _run_pipeline(
    job_id:         str,
    url:            Optional[str],
    upload_path:    Optional[str],
    num_highlights: int,
):
    def progress(stage: str, pct: int, msg: str):
        patch_job(job_id, status="processing", stage=stage,
                  progress=pct, message=msg)

    try:
        proc    = get_processor()
        results = proc.process(
            job_id            = job_id,
            source_url        = url,
            source_path       = upload_path,
            result_dir        = str(RESULTS_DIR / job_id),
            num_highlights    = num_highlights,
            progress_callback = progress,
        )
        patch_job(job_id,
                  status   = "done",
                  stage    = "done",
                  progress = 100,
                  message  = f"Done! {len(results)} highlight(s) ready.",
                  results  = results)

    except Exception as exc:
        logging.getLogger("maqta3").error("Job %s failed: %s", job_id, exc, exc_info=True)
        patch_job(job_id,
                  status  = "error",
                  message = f"Error: {exc}",
                  error   = str(exc))
