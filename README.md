# Segment Anything 2 Server

A server that runs [Meta's Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) via a REST API, designed for integration with [ScanXm](https://scanxm.com/), a visual segmentation tool.

---

## ✅ Features

- Full support for **SAM 2.1** models (Tiny, Small, Base+, Large)
- Compatible with **ScanXm** and other HTTP-based tools
- Optimized for **Windows with CUDA**, with fallback to CPU
- Custom `init_state()` bypasses video decoding and accepts raw pixel frames

---

## 🖥️ System Requirements

- Python **3.10**
- PyTorch **2.1+** with CUDA (e.g. CUDA 12.4)
- Conda (Miniconda recommended)

---

## 📦 Installation Instructions

### 1. Install Miniconda
Download and install [Miniconda for Windows 64-bit](https://docs.anaconda.com/miniconda/).

### 2. Create and activate a Conda environment
```bash
conda create --name sam2 python=3.10
conda activate sam2
```

### 3. Install PyTorch with CUDA
Go to [PyTorch's installation page](https://pytorch.org/get-started/locally/) and use the install command for your setup. Example:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 4. Clone and install SAM2
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

### 5. Download this server
Download `sam2_server.py` from:
[https://github.com/TristanWhitmarsh/segment-anything-2-server](https://github.com/TristanWhitmarsh/segment-anything-2-server)

Place it inside the `sam2` directory (where `setup.py` is located).

### 6. Install Flask
```bash
pip install flask
```

### 7. Download Checkpoints
Download the following model checkpoints and place them in a `checkpoints/` folder inside `sam2`:

- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

Checkpoints are available from the [official SAM2 repo](https://github.com/facebookresearch/sam2).

---

## 🚀 Running the Server

```bash
python sam2_server.py
```

Default URL:
```
http://localhost:8000
```

---

## 🔌 API Endpoints

| Endpoint       | Method | Description                                                     |
|----------------|--------|-----------------------------------------------------------------|
| `/init`        | POST   | Initialize volume inference from raw frames by mimicking video  |
| `/infer`       | POST   | Segment a frame using points or box                             |
| `/infer3D`     | POST   | Track objects across video frames                               |
| `/add_mask`    | POST   | Add binary mask to a specific frame                             |
| `/reset`       | POST   | Clear current inference state                                   |
| `/init2D`      | POST   | Initialize inference for single images                          |
| `/infer2D`     | POST   | Segment an individual 2D image                                  |

---

## 🛠️ Optional: Fix for OpenMP Errors on Windows

If you encounter an OpenMP runtime error such as:
```
OMP: Error #15: Initializing libiomp5md.dll...
```
Add the following lines at the **top of `sam2_server.py`**:

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

⚠️ This is a Windows-specific workaround for PyTorch/NumPy compatibility. Avoid using it in production or on Linux/Docker environments.

---

## 🧠 Using with ScanXm

Once the server is running, open **ScanXm** and go to the **Segment** tab:

1. Leave the default URL unless changed manually in the code.
2. Select a **model** from the dropdown (Tiny is fastest).
3. Use **Show Box** to constrain segmentation (optional).
4. Press **Initialize Segment Anything** to begin.
5. Use the tools:
   - **Box tool** to select region (LMB to add, RMB to remove)
   - **Point tool** to mark in/out points
   - **Add mask** to use existing labels as input
   - **Track volume** to propagate labels across 3D
   - **Reset** to clear state
6. To stop SAM2, press the **Initialize Segment Anything** button again.

🔎 SAM2 resizes inputs to 1024×1024, so results depend on original resolution.

---

## 📜 License

This project is licensed under the Apache License 2.0.
See [LICENSE](LICENSE) for details.

Segment Anything 2 is also licensed under Apache 2.0 by Meta.

---

## 🤝 Acknowledgements

- [Meta AI](https://ai.facebook.com/) for Segment Anything 2
- [ScanXm](https://scanxm.com/) for UI integration and testing

---

For help or bug reports, visit the [ScanXm SAM2 page](https://scanxm.com/segment-anything-2) or [GitHub issues](https://github.com/TristanWhitmarsh/segment-anything-2-server/issues).
