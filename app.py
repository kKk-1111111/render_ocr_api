import json
import multiprocessing
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl
from pypdf import PdfReader


APP_NAME = "Render OCR API"
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "./storage")).resolve()
API_TOKEN = os.getenv("API_TOKEN", "")
MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "100"))
MAX_PAGES_PER_JOB = int(os.getenv("MAX_PAGES_PER_JOB", "400"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "600"))
DEFAULT_DPI = int(os.getenv("DEFAULT_DPI", "150"))
DEFAULT_LANGS = os.getenv("DEFAULT_LANGS", "eng+chi_tra")
TEXT_THRESHOLD = int(os.getenv("TEXT_THRESHOLD", "30"))
PDF_TEXT_TIMEOUT_SECONDS = int(os.getenv("PDF_TEXT_TIMEOUT_SECONDS", "45"))
PAGE_RENDER_TIMEOUT_SECONDS = int(os.getenv("PAGE_RENDER_TIMEOUT_SECONDS", "90"))
PAGE_OCR_TIMEOUT_SECONDS = int(os.getenv("PAGE_OCR_TIMEOUT_SECONDS", "120"))
TESSERACT_TIMEOUT_SECONDS = int(os.getenv("TESSERACT_TIMEOUT_SECONDS", "120"))
OCR_WORKER_RECYCLE_PAGES = int(os.getenv("OCR_WORKER_RECYCLE_PAGES", "50"))
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=APP_NAME,
    version="1.0.0",
    description="OCR/text extraction API for Coze workflows and Render deployment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


JobStatus = Literal["queued", "processing", "completed", "completed_with_errors", "failed"]
OcrEngine = Literal["auto", "rapidocr", "tesseract"]


class JobRequest(BaseModel):
    file_url: HttpUrl = Field(..., description="Publicly reachable PDF URL.")
    file_name: Optional[str] = Field(None, description="Optional display file name.")
    max_pages: Optional[int] = Field(None, ge=1, le=2000, description="Optional per-job page limit.")
    page_start: Optional[int] = Field(None, ge=1, description="Optional first page to process, 1-based.")
    page_end: Optional[int] = Field(None, ge=1, description="Optional last page to process, inclusive.")
    dpi: int = Field(DEFAULT_DPI, ge=72, le=300, description="Rasterization DPI for scanned pages.")
    langs: str = Field(DEFAULT_LANGS, description="OCR language expression, e.g. eng+chi_tra.")
    ocr_engine: OcrEngine = Field("auto", description="OCR engine preference.")


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str
    result_url: str


class PageResult(BaseModel):
    page_number: int
    method: Literal["text", "rapidocr", "tesseract"]
    text: str
    confidence: Optional[float] = None
    dpi: Optional[int] = None
    elapsed_seconds: Optional[float] = None


class PageFailure(BaseModel):
    page_number: int
    stage: str
    error: str
    elapsed_seconds: Optional[float] = None


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    file_name: str
    total_pages: int = 0
    processed_pages: int = 0
    extracted_pages: int = 0
    failed_pages: int = 0
    text: str = ""
    pages: list[PageResult] = Field(default_factory=list)
    failures: list[PageFailure] = Field(default_factory=list)
    error: Optional[str] = None


def check_auth(authorization: Optional[str]) -> None:
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def safe_filename(name: str) -> str:
    name = name.rsplit("/", 1)[-1].strip() or "document.pdf"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


def job_dir(job_id: str) -> Path:
    return STORAGE_DIR / job_id


def job_json_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_job(job_id: str) -> dict[str, Any]:
    try:
        return json.loads(job_json_path(job_id).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


def save_job(data: dict[str, Any]) -> None:
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(job_json_path(data["job_id"]), data)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def elapsed_since(started_at: Any) -> Optional[float]:
    if not isinstance(started_at, (int, float)):
        return None
    return round(max(0.0, time.time() - started_at), 2)


def dependencies() -> dict[str, Any]:
    rapidocr_available = False
    rapidocr_import = None
    try:
        import importlib.util

        rapidocr_available = (
            importlib.util.find_spec("rapidocr_onnxruntime") is not None
            or importlib.util.find_spec("rapidocr") is not None
        )
    except Exception as exc:
        rapidocr_import = str(exc)

    return {
        "pdftotext": shutil.which("pdftotext"),
        "pdftoppm": shutil.which("pdftoppm"),
        "pdfinfo": shutil.which("pdfinfo"),
        "tesseract": shutil.which("tesseract"),
        "rapidocr": rapidocr_available,
        "rapidocr_import_error": rapidocr_import,
        "storage_dir": str(STORAGE_DIR),
        "max_download_mb": MAX_DOWNLOAD_MB,
        "max_pages_per_job": MAX_PAGES_PER_JOB,
        "default_dpi": DEFAULT_DPI,
        "text_threshold": TEXT_THRESHOLD,
        "pdf_text_timeout_seconds": PDF_TEXT_TIMEOUT_SECONDS,
        "page_render_timeout_seconds": PAGE_RENDER_TIMEOUT_SECONDS,
        "page_ocr_timeout_seconds": PAGE_OCR_TIMEOUT_SECONDS,
        "tesseract_timeout_seconds": TESSERACT_TIMEOUT_SECONDS,
        "ocr_worker_recycle_pages": OCR_WORKER_RECYCLE_PAGES,
        "cors_allow_origins": CORS_ALLOW_ORIGINS,
    }


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            returncode=124,
            stdout=str(exc.stdout or ""),
            stderr=f"command timed out after {timeout}s",
        )


def download_pdf(file_url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    downloaded = 0
    try:
        with requests.get(file_url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT_SECONDS)) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
                raise ValueError(f"URL does not look like a PDF: {content_type}")

            with output_path.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > limit:
                        raise ValueError(f"PDF is larger than MAX_DOWNLOAD_MB={MAX_DOWNLOAD_MB}")
                    file_handle.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"download failed: {exc}") from exc


async def save_uploaded_pdf(file: UploadFile, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content_type = (file.content_type or "").lower()
    if content_type and "pdf" not in content_type and "octet-stream" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file does not look like a PDF: {file.content_type}",
        )

    limit = MAX_DOWNLOAD_MB * 1024 * 1024
    total = 0
    try:
        with output_path.open("wb") as file_handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is larger than MAX_DOWNLOAD_MB={MAX_DOWNLOAD_MB}",
                    )
                file_handle.write(chunk)
    finally:
        await file.close()

    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return total


def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as exc:
        raise RuntimeError(f"failed to read PDF page count: {exc}") from exc


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_text_page(pdf_path: Path, page_number: int) -> tuple[str, Optional[str]]:
    if not shutil.which("pdftotext"):
        return "", "pdftotext is not installed"

    result = run_command(
        [
            "pdftotext",
            "-enc",
            "UTF-8",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            str(pdf_path),
            "-",
        ],
        timeout=PDF_TEXT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return "", (result.stderr or result.stdout or "pdftotext failed").strip()
    return result.stdout.strip(), None


def render_page(pdf_path: Path, page_number: int, images_dir: Path, dpi: int) -> Path:
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm is not installed")

    images_dir.mkdir(parents=True, exist_ok=True)
    prefix = images_dir / f"page_{page_number:04d}"
    result = run_command(
        [
            "pdftoppm",
            "-gray",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ],
        timeout=PAGE_RENDER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "pdftoppm failed").strip())
    image_path = prefix.with_suffix(".png")
    if not image_path.exists():
        raise RuntimeError(f"pdftoppm did not produce image: {image_path}")
    return image_path


_rapidocr_instance: Any = None


def rapidocr_image(image_path: Path) -> tuple[str, Optional[float]]:
    global _rapidocr_instance
    try:
        if _rapidocr_instance is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                from rapidocr import RapidOCR

            _rapidocr_instance = RapidOCR()

        raw = _rapidocr_instance(str(image_path))
        rows = raw[0] if isinstance(raw, tuple) else raw

        if hasattr(rows, "txts"):
            texts = [str(item) for item in getattr(rows, "txts", []) if item]
            scores = [float(item) for item in getattr(rows, "scores", []) if item is not None]
            confidence = sum(scores) / len(scores) if scores else None
            return "\n".join(texts).strip(), confidence

        texts: list[str] = []
        scores: list[float] = []
        for item in rows or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                texts.append(str(item[1]))
                if len(item) >= 3:
                    try:
                        scores.append(float(item[2]))
                    except Exception:
                        pass
        confidence = sum(scores) / len(scores) if scores else None
        return "\n".join(texts).strip(), confidence
    except Exception as exc:
        raise RuntimeError(f"RapidOCR failed: {exc}") from exc


def tesseract_image(image_path: Path, langs: str) -> str:
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract is not installed")
    result = run_command(
        ["tesseract", str(image_path), "stdout", "-l", langs, "--psm", "6"],
        timeout=TESSERACT_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "tesseract failed").strip())
    return result.stdout.strip()


def ocr_rendered_image(
    image_path: Path,
    engine: OcrEngine,
    langs: str,
) -> tuple[str, str, Optional[float]]:
    if engine == "tesseract":
        engine_order = ["tesseract", "rapidocr"]
    elif engine == "rapidocr":
        engine_order = ["rapidocr", "tesseract"]
    else:
        engine_order = ["rapidocr", "tesseract"]

    errors: list[str] = []
    deps = dependencies()
    for candidate in engine_order:
        if candidate == "rapidocr":
            if not deps["rapidocr"]:
                errors.append("rapidocr is not installed")
                continue
            try:
                text, confidence = rapidocr_image(image_path)
                if normalize_text(text):
                    return text, "rapidocr", confidence
                errors.append("RapidOCR returned empty text")
            except Exception as exc:
                errors.append(str(exc))
            continue

        if candidate == "tesseract":
            if not shutil.which("tesseract"):
                errors.append("tesseract is not installed")
                continue
            try:
                text = tesseract_image(image_path, langs)
                if normalize_text(text):
                    return text, "tesseract", None
                errors.append("Tesseract returned empty text")
            except Exception as exc:
                errors.append(str(exc))

    raise RuntimeError("; ".join(errors) or "No OCR engine available.")


def ocr_page(
    pdf_path: Path,
    page_number: int,
    images_dir: Path,
    engine: OcrEngine,
    dpi: int,
    langs: str,
) -> tuple[str, str, Optional[float]]:
    image_path = render_page(pdf_path, page_number, images_dir, dpi)
    try:
        return ocr_rendered_image(image_path, engine, langs)
    finally:
        try:
            image_path.unlink(missing_ok=True)
        except Exception:
            pass


def _ocr_image_process_worker(
    queue: multiprocessing.Queue,
    image_path: str,
    engine: OcrEngine,
    langs: str,
) -> None:
    try:
        queue.put(("ok", ocr_rendered_image(Path(image_path), engine, langs)))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def ocr_image_with_timeout(
    image_path: Path,
    engine: OcrEngine,
    langs: str,
) -> tuple[str, str, Optional[float]]:
    queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(
        target=_ocr_image_process_worker,
        args=(queue, str(image_path), engine, langs),
    )
    process.start()
    process.join(PAGE_OCR_TIMEOUT_SECONDS)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(
            f"OCR timed out after {PAGE_OCR_TIMEOUT_SECONDS}s for image {image_path.name}"
        )

    if process.exitcode not in (0, None) and queue.empty():
        raise RuntimeError(f"OCR worker exited with code {process.exitcode}")

    if queue.empty():
        raise RuntimeError("OCR worker returned no result")

    status, payload = queue.get_nowait()
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def process_job(job_id: str, request: dict[str, Any]) -> None:
    data = load_job(job_id)
    try:
        data["status"] = "processing"
        data["stage"] = "starting"
        data["started_at"] = utc_now()
        data["started_at_epoch"] = time.time()
        data["dpi"] = request.get("dpi")
        data["langs"] = request.get("langs")
        data["ocr_engine"] = request.get("ocr_engine")
        data["page_render_timeout_seconds"] = PAGE_RENDER_TIMEOUT_SECONDS
        data["page_ocr_timeout_seconds"] = PAGE_OCR_TIMEOUT_SECONDS
        data["pdf_text_timeout_seconds"] = PDF_TEXT_TIMEOUT_SECONDS
        save_job(data)

        job_path = job_dir(job_id)
        pdf_path = job_path / "input.pdf"
        images_dir = job_path / "images"
        if request.get("uploaded_pdf_path"):
            data["stage"] = "using_uploaded_file"
            if not pdf_path.exists():
                raise RuntimeError("uploaded PDF file is missing")
            save_job(data)
        else:
            data["stage"] = "downloading"
            save_job(data)
            download_pdf(request["file_url"], pdf_path)

        data["stage"] = "reading_pdf"
        total_pages = get_pdf_page_count(pdf_path)
        data["total_pages"] = total_pages
        save_job(data)

        max_pages = request.get("max_pages") or MAX_PAGES_PER_JOB
        page_start = request.get("page_start") or 1
        page_end = request.get("page_end") or total_pages
        page_end = min(page_end, total_pages)
        if page_start > page_end:
            raise RuntimeError("page_start must be <= page_end")

        selected_pages = page_end - page_start + 1
        if selected_pages > max_pages:
            raise RuntimeError(
                f"PDF page range has {selected_pages} pages, exceeds max_pages={max_pages}. "
                "Split the PDF or increase max_pages."
            )

        deps = dependencies()
        if not deps["pdftotext"] and not deps["rapidocr"] and not deps["tesseract"]:
            raise RuntimeError(
                "No extraction method available. Install poppler-utils and at least one OCR engine."
            )

        pages: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for page_number in range(page_start, page_end + 1):
            page_started_at = time.time()
            image_path: Optional[Path] = None
            failure_stage = "text_extract"
            try:
                data["stage"] = f"processing_page_{page_number}_text_extract"
                data["current_page"] = page_number
                data["current_page_started_at"] = utc_now()
                data["current_page_started_at_epoch"] = page_started_at
                data["current_page_elapsed_seconds"] = 0
                data["processed_pages"] = page_number - page_start
                save_job(data)

                text, text_error = extract_text_page(pdf_path, page_number)
                if len(normalize_text(text)) >= TEXT_THRESHOLD:
                    elapsed = elapsed_since(page_started_at)
                    pages.append(
                        {
                            "page_number": page_number,
                            "method": "text",
                            "text": text,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    full_text_parts.append(f"\n\n--- Page {page_number} ---\n{text}")
                    data["stage"] = f"processing_page_{page_number}_done"
                    continue

                failure_stage = "render"
                data["stage"] = f"processing_page_{page_number}_render"
                data["current_page_elapsed_seconds"] = elapsed_since(page_started_at)
                save_job(data)
                image_path = render_page(pdf_path, page_number, images_dir, request["dpi"])

                failure_stage = "ocr"
                data["stage"] = f"processing_page_{page_number}_ocr"
                data["current_page_elapsed_seconds"] = elapsed_since(page_started_at)
                save_job(data)
                ocr_text, method, confidence = ocr_image_with_timeout(
                    image_path=image_path,
                    engine=request["ocr_engine"],
                    langs=request["langs"],
                )

                if normalize_text(ocr_text):
                    page_result = {
                        "page_number": page_number,
                        "method": method,
                        "text": ocr_text,
                        "dpi": request["dpi"],
                        "elapsed_seconds": elapsed_since(page_started_at),
                    }
                    if confidence is not None:
                        page_result["confidence"] = confidence
                    pages.append(page_result)
                    full_text_parts.append(f"\n\n--- Page {page_number} ---\n{ocr_text}")
                    data["stage"] = f"processing_page_{page_number}_done"
                else:
                    data["stage"] = f"processing_page_{page_number}_failed"
                    failures.append(
                        {
                            "page_number": page_number,
                            "stage": "ocr",
                            "error": text_error or "OCR returned empty text",
                            "elapsed_seconds": elapsed_since(page_started_at),
                        }
                    )
            except Exception as exc:
                data["stage"] = f"processing_page_{page_number}_failed"
                failures.append(
                    {
                        "page_number": page_number,
                        "stage": failure_stage,
                        "error": str(exc),
                        "elapsed_seconds": elapsed_since(page_started_at),
                    }
                )
            finally:
                if image_path is not None:
                    try:
                        image_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                data["processed_pages"] = page_number - page_start + 1
                data["extracted_pages"] = len(pages)
                data["failed_pages"] = len(failures)
                data["pages"] = pages
                data["failures"] = failures
                data["text"] = "".join(full_text_parts).strip()
                data["last_page_elapsed_seconds"] = elapsed_since(page_started_at)
                data["current_page_elapsed_seconds"] = elapsed_since(page_started_at)
                save_job(data)

        data["processed_pages"] = selected_pages
        data["extracted_pages"] = len(pages)
        data["failed_pages"] = len(failures)
        data["pages"] = pages
        data["failures"] = failures
        data["text"] = "".join(full_text_parts).strip()
        data["stage"] = "done"
        data["current_page"] = None
        data["current_page_started_at"] = None
        data["current_page_started_at_epoch"] = None
        data["current_page_elapsed_seconds"] = None
        data["elapsed_seconds"] = elapsed_since(data.get("started_at_epoch"))

        if pages and failures:
            data["status"] = "completed_with_errors"
        elif pages:
            data["status"] = "completed"
        else:
            data["status"] = "failed"
            data["error"] = f"All {selected_pages} pages failed to extract text."
        save_job(data)
    except Exception as exc:
        data["status"] = "failed"
        data["error"] = str(exc)
        data["stage"] = "failed"
        save_job(data)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "app": APP_NAME}


@app.get("/dependencies")
def dependency_status() -> dict[str, Any]:
    return dependencies()


@app.post("/jobs", response_model=JobCreateResponse)
def create_job(
    payload: JobRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
) -> JobCreateResponse:
    check_auth(authorization)
    job_id = uuid.uuid4().hex
    file_name = safe_filename(payload.file_name or str(payload.file_url))
    data: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_name": file_name,
        "file_url": str(payload.file_url),
        "total_pages": 0,
        "processed_pages": 0,
        "extracted_pages": 0,
        "failed_pages": 0,
        "pages": [],
        "failures": [],
        "text": "",
        "error": None,
        "dpi": payload.dpi,
        "langs": payload.langs,
        "ocr_engine": payload.ocr_engine,
        "current_page": None,
        "current_page_started_at": None,
        "current_page_started_at_epoch": None,
        "current_page_elapsed_seconds": None,
        "page_render_timeout_seconds": PAGE_RENDER_TIMEOUT_SECONDS,
        "page_ocr_timeout_seconds": PAGE_OCR_TIMEOUT_SECONDS,
        "pdf_text_timeout_seconds": PDF_TEXT_TIMEOUT_SECONDS,
    }
    save_job(data)
    background_tasks.add_task(process_job, job_id, payload.model_dump(mode="json"))
    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
        result_url=f"/jobs/{job_id}/result",
    )


@app.post("/jobs/upload", response_model=JobCreateResponse)
async def create_upload_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_name: Optional[str] = Form(default=None),
    max_pages: Optional[int] = Form(default=None),
    page_start: Optional[int] = Form(default=None),
    page_end: Optional[int] = Form(default=None),
    dpi: int = Form(default=DEFAULT_DPI),
    langs: str = Form(default=DEFAULT_LANGS),
    ocr_engine: OcrEngine = Form(default="auto"),
    authorization: Optional[str] = Header(default=None),
) -> JobCreateResponse:
    check_auth(authorization)
    if max_pages is not None and max_pages < 1:
        raise HTTPException(status_code=400, detail="max_pages must be >= 1")
    if page_start is not None and page_start < 1:
        raise HTTPException(status_code=400, detail="page_start must be >= 1")
    if page_end is not None and page_end < 1:
        raise HTTPException(status_code=400, detail="page_end must be >= 1")
    if dpi < 72 or dpi > 300:
        raise HTTPException(status_code=400, detail="dpi must be between 72 and 300")

    job_id = uuid.uuid4().hex
    job_path = job_dir(job_id)
    pdf_path = job_path / "input.pdf"
    display_name = safe_filename(file_name or file.filename or "uploaded.pdf")
    uploaded_bytes = await save_uploaded_pdf(file, pdf_path)

    request_data: dict[str, Any] = {
        "uploaded_pdf_path": str(pdf_path),
        "file_name": display_name,
        "max_pages": max_pages,
        "page_start": page_start,
        "page_end": page_end,
        "dpi": dpi,
        "langs": langs,
        "ocr_engine": ocr_engine,
    }
    data: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "file_name": display_name,
        "file_url": None,
        "uploaded_bytes": uploaded_bytes,
        "total_pages": 0,
        "processed_pages": 0,
        "extracted_pages": 0,
        "failed_pages": 0,
        "pages": [],
        "failures": [],
        "text": "",
        "error": None,
        "dpi": dpi,
        "langs": langs,
        "ocr_engine": ocr_engine,
        "current_page": None,
        "current_page_started_at": None,
        "current_page_started_at_epoch": None,
        "current_page_elapsed_seconds": None,
        "page_render_timeout_seconds": PAGE_RENDER_TIMEOUT_SECONDS,
        "page_ocr_timeout_seconds": PAGE_OCR_TIMEOUT_SECONDS,
        "pdf_text_timeout_seconds": PDF_TEXT_TIMEOUT_SECONDS,
    }
    save_job(data)
    background_tasks.add_task(process_job, job_id, request_data)
    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
        result_url=f"/jobs/{job_id}/result",
    )


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
    check_auth(authorization)
    data = load_job(job_id)
    current_page_elapsed = data.get("current_page_elapsed_seconds")
    if data.get("status") == "processing":
        current_page_elapsed = elapsed_since(data.get("current_page_started_at_epoch"))
    elapsed_seconds = data.get("elapsed_seconds")
    if data.get("started_at_epoch"):
        elapsed_seconds = elapsed_since(data.get("started_at_epoch"))
    return {
        "job_id": data["job_id"],
        "status": data["status"],
        "stage": data.get("stage"),
        "file_name": data.get("file_name"),
        "total_pages": data.get("total_pages", 0),
        "processed_pages": data.get("processed_pages", 0),
        "extracted_pages": data.get("extracted_pages", 0),
        "failed_pages": data.get("failed_pages", 0),
        "error": data.get("error"),
        "updated_at": data.get("updated_at"),
        "dpi": data.get("dpi"),
        "langs": data.get("langs"),
        "ocr_engine": data.get("ocr_engine"),
        "current_page": data.get("current_page"),
        "current_page_started_at": data.get("current_page_started_at"),
        "current_page_elapsed_seconds": current_page_elapsed,
        "elapsed_seconds": elapsed_seconds,
        "last_page_elapsed_seconds": data.get("last_page_elapsed_seconds"),
        "page_render_timeout_seconds": data.get(
            "page_render_timeout_seconds", PAGE_RENDER_TIMEOUT_SECONDS
        ),
        "page_ocr_timeout_seconds": data.get(
            "page_ocr_timeout_seconds", PAGE_OCR_TIMEOUT_SECONDS
        ),
        "pdf_text_timeout_seconds": data.get(
            "pdf_text_timeout_seconds", PDF_TEXT_TIMEOUT_SECONDS
        ),
    }


@app.get("/jobs/{job_id}/result", response_model=JobResult)
def get_job_result(job_id: str, authorization: Optional[str] = Header(default=None)) -> JobResult:
    check_auth(authorization)
    data = load_job(job_id)
    if data["status"] in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Job is not complete yet")
    return JobResult(**data)


@app.get("/jobs/{job_id}/text", response_class=PlainTextResponse)
def get_job_text(job_id: str, authorization: Optional[str] = Header(default=None)) -> str:
    check_auth(authorization)
    data = load_job(job_id)
    if data["status"] in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Job is not complete yet")
    return data.get("text") or ""
