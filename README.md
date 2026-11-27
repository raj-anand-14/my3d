# my3d — your own image-to-3D generator (TripoSG starter)

Image in → 3D mesh (GLB) out, served from your own GPU with a small web UI.
Built on **TripoSG** (open-sourced by Tripo / VAST AI Research, MIT license).

## What you need
- NVIDIA GPU with **8GB+ VRAM** (12–24GB is more comfortable)
- Linux or Windows + WSL2, Python 3.10, git
- No GPU? Rent one (RunPod, Lambda, vast.ai) or use Colab and run the same steps.

## Steps

```bash
# 1. Python env
conda create -n my3d python=3.10 -y
conda activate my3d

# 2. Install (set CUDA_TAG to match `nvidia-smi`, e.g. cu118 / cu121 / cu124)
CUDA_TAG=cu121 bash setup.sh

# 3. Run. Weights download automatically on first start
python server.py
```

Then open **http://localhost:8000**, upload a photo of one object, and click *Generate 3D*.

From the command line:
```bash
curl -F image=@chair.png -F faces=20000 http://localhost:8000/generate -o chair.glb
```

## Troubleshooting
- **`diso` fails to build:** install the CUDA toolkit so `nvcc` exists
  (`conda install -c nvidia cuda-toolkit`), then rerun `pip install --no-build-isolation diso`.
- **"Unable to load plugins … libOpenGL.so.0" warnings:** harmless on headless servers.
  To silence them: `sudo apt install libopengl0`.
- **CUDA out of memory:** lower `steps`, close other GPU apps, or use a bigger GPU.

## How it works
1. **Background removal** (RMBG-1.4) cuts the object out and puts it on white.
2. **TripoSG**, a 1.5B-parameter rectified-flow transformer, generates an SDF shape from the image.
3. The mesh is extracted, simplified to `faces` triangles (pymeshlab) and exported as GLB.

## Knobs
| Param    | Default | Effect |
|----------|---------|--------|
| seed     | 42      | Different seed, different variation |
| steps    | 50      | Fewer = faster, rougher |
| guidance | 7.0     | How strongly the output follows the image |
| faces    | 20000   | Lower = lighter mesh (games/web), higher = more detail |

## Input tips
Use one object, centered, fully in frame, with even lighting. Product shots and
concept-art renders work best. Busy scenes and partly hidden objects work worst.

## Known limits / next steps
- **Geometry only, no texture.** TripoSG outputs shape. For color/PBR textures,
  add MV-Adapter (from the same team) or switch the backend to TRELLIS.2 / Hunyuan3D.
- **Text-to-3D:** put a text-to-image model (e.g. FLUX / SDXL) in front: prompt → image → this server.
- **Licensing:** TripoSG is MIT, but **RMBG-1.4 is non-commercial**. Swap in
  `rembg` (u2net) or BiRefNet before selling anything.
- **Scaling:** for multiple users, add a job queue (Redis + RQ/Celery) instead of the single GPU lock.
- **Fine-tuning:** to specialize (game props, furniture, characters) you'd train on your
  own image+mesh pairs. Pixal3D ships training code if you go that route.

## Credits
- [TripoSG](https://github.com/VAST-AI-Research/TripoSG) by VAST AI Research (MIT), the 3D generation model this wraps
- [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) by BRIA AI (non-commercial license), background removal
- [model-viewer](https://modelviewer.dev/) by Google, the in-browser 3D viewer
