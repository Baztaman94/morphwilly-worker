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
import tempfile
import traceback
import urllib.request

import runpod

FACEFUSION_DIR = os.environ.get("FACEFUSION_DIR", "/app/facefusion")


def _download(url: str, dest: pathlib.Path) -> None:
    with urllib.request.urlopen(url, timeout=180) as r:  # noqa: S310
        dest.write_bytes(r.read())


def _run_facefusion(face: pathlib.Path, target: pathlib.Path, out: pathlib.Path) -> None:
    # Adapt flags to your installed FaceFusion version if needed.
    cmd = [
        "python", "facefusion.py", "headless-run",
        "--source-paths", str(face),
        "--target-path", str(target),
        "--output-path", str(out),
        "--execution-providers", "cuda",
    ]
    subprocess.run(cmd, cwd=FACEFUSION_DIR, check=True)


def _maybe_lip_sync(video: pathlib.Path, options: dict) -> pathlib.Path:
    # Placeholder extension point: run Wav2Lip / LatentSync here when
    # options.get("lipSync") and an audio track is provided. Returns the
    # (possibly new) video path. No-op by default.
    return video


def _deliver(out: pathlib.Path) -> dict:
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
            _download(source_url, target)
            _download(face_url, face)

            _run_facefusion(face, target, out)
            out = _maybe_lip_sync(out, options)

            return _deliver(out)
    except subprocess.CalledProcessError as e:
        return {"error": f"FaceFusion a échoué (code {e.returncode})."}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}


runpod.serverless.start({"handler": handler})
