"""my3d — a tiny "own Tripo" server: image in, GLB out.

Loads TripoSG + background removal ONCE at startup, then serves:
  GET  /           -> web page (upload an image, view the 3D result)
  POST /generate   -> multipart image upload, returns a .glb file
"""
import io
import os
import sys
import threading
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parent
TRIPOSG_DIR = ROOT / "TripoSG"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Make TripoSG's package and its scripts/ helpers importable
sys.path.insert(0, str(TRIPOSG_DIR))
sys.path.insert(0, str(TRIPOSG_DIR / "scripts"))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: E402
from briarmbg import BriaRMBG  # noqa: E402
from inference_triposg import run_triposg  # noqa: E402

DEVICE = "cuda"
DTYPE = torch.float16

if not torch.cuda.is_available():
    sys.exit("No CUDA GPU found. TripoSG needs an NVIDIA GPU with 8GB+ VRAM.")

# ---- load models once -------------------------------------------------------
weights = TRIPOSG_DIR / "pretrained_weights"
triposg_dir = weights / "TripoSG"
rmbg_dir = weights / "RMBG-1.4"
print("Downloading/loading weights (first run takes a while)...")
snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=str(triposg_dir))
snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=str(rmbg_dir))

rmbg_net = BriaRMBG.from_pretrained(str(rmbg_dir)).to(DEVICE).eval()
pipe = TripoSGPipeline.from_pretrained(str(triposg_dir)).to(DEVICE, DTYPE)
gpu_lock = threading.Lock()  # one generation at a time on the GPU
print("Models ready.")

# ---- web app ----------------------------------------------------------------
app = FastAPI(title="my3d")


@app.post("/generate")
def generate(
    image: UploadFile = File(...),
    seed: int = Form(42),
    steps: int = Form(50),
    guidance: float = Form(7.0),
    faces: int = Form(20000),
):
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGBA")
    except Exception:
        raise HTTPException(400, "Could not read that image.")

    job = uuid.uuid4().hex
    in_path = OUT_DIR / f"{job}.png"   # TripoSG's prepare_image expects a file path
    img.save(in_path)
    out_path = OUT_DIR / f"{job}.glb"
    with gpu_lock:
        mesh = run_triposg(
            pipe,
            image_input=str(in_path),
            rmbg_net=rmbg_net,
            seed=seed,
            num_inference_steps=steps,
            guidance_scale=guidance,
            faces=faces,
        )
    mesh.export(str(out_path))
    return FileResponse(out_path, media_type="model/gltf-binary", filename=out_path.name)


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
