import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import modal

APP_NAME = "wonder3d-plus-infer"
VOLUME_NAME = "wonder3d-plus-cache"
MOUNT = "/models"

ROOT = Path(__file__).parent  # repo root

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# NOTE: copy=True lets us run build steps AFTER adding local files.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "libgl1", "libglib2.0-0", "build-essential", "python3-dev", "cmake",
        "libegl1", "libgles2", "libglvnd0", "mesa-utils", "libosmesa6"
    )
    .env({
        "PYTHONPATH": "/opt/Wonder3D",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0+PTX",
        "FORCE_CUDA": "1",
        "CUDA_HOME": "/usr/local/cuda",
        "CC": "gcc",
        "CXX": "g++",
        "PYOPENGL_PLATFORM": "egl",
        "PYGLET_HEADLESS": "true",
    })
    .add_local_dir(str(ROOT), remote_path="/opt/Wonder3D", copy=True)
    .run_commands(
        "python -m pip install -U pip setuptools wheel",
        # avoid NumPy 2.x binary-compat issues with some ML wheels
        "python -m pip install numpy==1.26.4 pillow==10.4.0",
        # hf download helper (README uses snapshot_download)
        "python -m pip install huggingface_hub",
        # install repo deps
        "python -m pip install -r /opt/Wonder3D/requirements.txt",
        "python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@stable'",
        "python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git'",
        "python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html",
        "python -m pip install --no-build-isolation torch-efficient-distloss",
    )
)

app = modal.App(APP_NAME)

def _zip_dir(dir_path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in dir_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(dir_path)))
    return buf.getvalue()

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 30,
    volumes={MOUNT: volume},
)
def wonder3d_plus_run(
    image_bytes: bytes,
    camera_type: str = "ortho",   # 'ortho' or 'persp'
    crop_size: int = 192,         # README default is 192 (relative to 256 input)
    num_refine: int = 2,          # README default is 2
) -> bytes:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", f"{MOUNT}/hf")
    os.environ.setdefault("HF_HUB_CACHE", f"{MOUNT}/hf/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{MOUNT}/hf/transformers")

    ckpts_dir = Path(MOUNT) / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Download checkpoints once into the persistent volume (if missing).
    # README repo_id: flamehaze1115/Wonder3D_plus :contentReference[oaicite:3]{index=3}
    marker = ckpts_dir / ".download_complete"
    if not marker.exists():
        snapshot_download(repo_id="flamehaze1115/Wonder3D_plus", local_dir=str(ckpts_dir))
        marker.write_text("ok\n")
        volume.commit()  # persist to the volume

    repo_root = Path("/opt/Wonder3D")
    repo_ckpts = repo_root / "ckpts"
    if repo_ckpts.exists() and not repo_ckpts.is_symlink():
        # If a real folder exists, don't silently mix states.
        shutil.rmtree(repo_ckpts)
    if not repo_ckpts.exists():
        os.symlink(str(ckpts_dir), str(repo_ckpts))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / "input.png"
        out_path = td / "outputs"

        in_path.write_bytes(image_bytes)
        out_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "run.py",
            "--input_path",
            str(in_path),
            "--output_path",
            str(out_path),
            "--crop_size",
            str(crop_size),
            "--camera_type",
            str(camera_type),
            "--num_refine",
            str(num_refine),
        ]

        subprocess.run(cmd, cwd=str(repo_root), check=True)

        return _zip_dir(out_path)

@app.local_entrypoint()
def main(
    input_path: str,
    out_zip: str = "wonder3d_plus_out.zip",
    camera_type: str = "ortho",
    crop_size: int = 192,
    num_refine: int = 2,
):
    data = Path(input_path).read_bytes()
    out = wonder3d_plus_run.remote(
        data,
        camera_type=camera_type,
        crop_size=crop_size,
        num_refine=num_refine,
    )
    Path(out_zip).write_bytes(out)
    print(out_zip)
