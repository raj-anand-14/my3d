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

# `diso` gives fast GPU mesh extraction but needs a CUDA compile that can fail.
# If it's missing, stub it and use TripoSG's scikit-image extraction instead.
try:
    import diso  # noqa: F401
    HAS_DISO = True
except Exception:
    import types
    _stub = types.ModuleType("diso")
    _stub.DiffDMC = None
    sys.modules["diso"] = _stub
    HAS_DISO = False

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: E402
from briarmbg import BriaRMBG  # noqa: E402
from image_process import prepare_image  # noqa: E402
from inference_triposg import simplify_mesh  # noqa: E402

print(f"Mesh extraction: {'diso (fast GPU)' if HAS_DISO else 'scikit-image fallback (diso not installed)'}")


@torch.no_grad()
def run_triposg(pipe, image_input, rmbg_net, seed, num_inference_steps=50, guidance_scale=7.0, faces=-1):
    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_flash_decoder=HAS_DISO,
    ).samples[0]
    if outputs[0] is None:
        raise RuntimeError("Mesh extraction failed (no surface found). Try another image or seed.")
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
    if faces > 0:
        mesh = simplify_mesh(mesh, faces)
    return mesh

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
    try:
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
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    mesh.export(str(out_path))
    return FileResponse(out_path, media_type="model/gltf-binary", filename=out_path.name)


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
