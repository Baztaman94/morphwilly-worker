"""RunPod Serverless handler for MorphWilly face replacement.

Input (job["input"]):
  { "sourceUrl": str, "faceImageUrl": str, "options": { "lipSync": bool } }

Output:
  { "outputUrl": str }        if S3 storage is configured (recommended)
  { "outputBase64": str,
    "filename": str }         otherwise (short clips only — RunPod payload cap)
  { "error": str }            on failure

The consent gate lives in the MorphWilly app, upstream of this worker: the app
never submits a job for a real person's likeness without a verified licence.
This worker only ever runs inside that flow.
"""

import base64
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback
import urllib.request

import requests
import runpod

FACEFUSION_DIR = os.environ.get("FACEFUSION_DIR", "/app/facefusion")


def log(*args) -> None:
    """Print to stdout, flushed, so RunPod captures it in the Container logs."""
    print(*args, flush=True)


def _download(url: str, dest: pathlib.Path) -> None:
    with urllib.request.urlopen(url, timeout=180) as r:  # noqa: S310
        dest.write_bytes(r.read())


def _run_facefusion(
    face: pathlib.Path, target: pathlib.Path, out: pathlib.Path, quality: str
) -> None:
    # Two presets driven by the project's quality setting:
    #   draft → fast preview (swap only, low pixel boost, lighter encode)
    #   high  → delivery (swap + GFPGAN restoration, 512px, high quality encode)
    base = [
        "python", "facefusion.py", "headless-run",
        "--face-selector-mode", "many",
    ]
    if quality == "draft":
        preset = [
            "--processors", "face_swapper",
            "--face-swapper-pixel-boost", "128x128",
            "--output-video-quality", "80",
        ]
    else:  # high
        preset = [
            "--processors", "face_swapper", "face_enhancer",
            "--face-swapper-pixel-boost", "512x512",
            "--face-enhancer-model", "gfpgan_1.4",
            "--face-enhancer-blend", "80",
            "--output-video-quality", "95",
        ]
    cmd = base + preset + [
        "--source-paths", str(face),
        "--target-path", str(target),
        "--output-path", str(out),
        "--execution-providers", "cuda",
    ]
    log(f"FaceFusion (qualité={quality}):", " ".join(cmd))
    # Stream FaceFusion's own output line-by-line into the container logs
    # (so we see progress live instead of only after it finishes).
    proc = subprocess.Popen(
        cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        line = line.rstrip()
        log("[ff]", line)
        if line:
            tail.append(line)
            tail[:] = tail[-12:]  # keep only the last 12 lines
    proc.wait()
    if proc.returncode != 0:
        # Surface FaceFusion's actual last words so the error shows in the app.
        raise RuntimeError(
            f"FaceFusion code {proc.returncode}. Dernières lignes:\n" + "\n".join(tail)
        )


def _log_gpu() -> None:
    """Report whether onnxruntime can actually use the GPU (CUDA)."""
    try:
        import onnxruntime as ort
        provs = ort.get_available_providers()
        log("onnxruntime providers:", provs)
        if "CUDAExecutionProvider" not in provs:
            log("⚠️ CUDAExecutionProvider ABSENT → FaceFusion tournera sur CPU (lent).")
    except Exception as e:  # noqa: BLE001
        log("onnxruntime check impossible:", e)


def _maybe_lip_sync(video: pathlib.Path, options: dict) -> pathlib.Path:
    # Placeholder extension point: run Wav2Lip / LatentSync here when
    # options.get("lipSync") and an audio track is provided. Returns the
    # (possibly new) video path. No-op by default.
    return video


def _deliver(out: pathlib.Path, options: dict) -> dict:
    # Preferred: push the finished video straight back to the app through the
    # tunnel, so large HQ files don't have to fit in RunPod's inline response.
    upload_url = options.get("uploadUrl")
    if upload_url:
        log("Renvoi de la vidéo vers l'app:", upload_url, f"({out.stat().st_size} o)")
        with open(out, "rb") as f:
            resp = requests.post(
                upload_url,
                data=f,
                headers={
                    "x-upload-secret": options.get("uploadSecret", ""),
                    "x-filename": out.name,
                    "Content-Type": "application/octet-stream",
                    "ngrok-skip-browser-warning": "1",
                },
                timeout=600,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"Renvoi refusé ({resp.status_code}): {resp.text[:200]}")
        url = resp.json().get("url")
        if not url:
            raise RuntimeError("Renvoi accepté mais sans url en retour.")
        return {"outputUrl": url}

    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        import boto3  # imported lazily so base64 mode needs no boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT") or None,
            region_name=os.environ.get("S3_REGION") or None,
        )
        key = f"renders/{out.name}"
        s3.upload_file(str(out), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
        public_base = os.environ.get("S3_PUBLIC_BASE_URL", "").rstrip("/")
        url = f"{public_base}/{key}" if public_base else s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86400
        )
        return {"outputUrl": url}

    # Fallback: inline base64 (short clips only).
    return {
        "outputBase64": base64.b64encode(out.read_bytes()).decode("ascii"),
        "filename": out.name,
    }


def handler(job: dict) -> dict:
    data = job.get("input", {}) or {}
    source_url = data.get("sourceUrl")
    face_url = data.get("faceImageUrl")
    options = data.get("options", {}) or {}

    if not source_url or not face_url:
        return {"error": "sourceUrl et faceImageUrl sont requis."}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpd = pathlib.Path(tmp)
            target = tmpd / "source.mp4"
            face = tmpd / "face.jpg"
            out = tmpd / "output.mp4"
            log("Téléchargement de la source:", source_url)
            _download(source_url, target)
            log("Téléchargement du visage:", face_url)
            _download(face_url, face)
            log(f"Source {target.stat().st_size} o, visage {face.stat().st_size} o")

            _log_gpu()
            quality = "draft" if options.get("quality") == "draft" else "high"
            log("Lancement de FaceFusion…")
            _run_facefusion(face, target, out, quality)
            out = _maybe_lip_sync(out, options)
            log("Rendu terminé:", out, out.stat().st_size, "octets")

            return _deliver(out, options)
    except subprocess.CalledProcessError as e:
        log("FaceFusion a échoué, code", e.returncode)
        return {"error": f"FaceFusion a échoué (code {e.returncode})."}
    except Exception as e:  # noqa: BLE001
        log("Erreur:", type(e).__name__, e)
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}


runpod.serverless.start({"handler": handler})
