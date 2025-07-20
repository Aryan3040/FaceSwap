import cv2
import numpy as np
import gradio as gr
import insightface

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from gfpgan import GFPGANer

# === Constants ===
MAX_DIM = 1200  # Max width/height before we downscale

# === Utility: Resize if image is too large ===
def resize_if_too_large(img_bgr, max_dim=MAX_DIM):
    """
    If either width or height > max_dim, downscale so the longest side = max_dim.
    Keeps aspect ratio using INTER_AREA for better downscaling quality.
    """
    h, w, _ = img_bgr.shape
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_h = int(h * scale)
        new_w = int(w * scale)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr

# === Setup InsightFace ===
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# === Load Face Swapper ===
swapper = get_model('inswapper_128', download=False)

# === Initialize GFPGAN ===
restorer = GFPGANer(
    model_path="./GFPGANv1.4.pth",
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None  # If you don't want Real-ESRGAN for background
)

def swap_faces(source_img, target_img):
    try:
        # Convert PIL -> OpenCV BGR
        src_bgr = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        # 1) Auto-resize if images are too large
        src_bgr = resize_if_too_large(src_bgr, MAX_DIM)
        dst_bgr = resize_if_too_large(dst_bgr, MAX_DIM)

        # 2) Face detection
        src_faces = app.get(src_bgr)
        dst_faces = app.get(dst_bgr)

        if not src_faces:
            return "❌ No face detected in source image", None
        if not dst_faces:
            return "❌ No face detected in target image", None

        # 3) Face swap (swapping the first detected src face into each detected target face)
        for face in dst_faces:
            dst_bgr = swapper.get(dst_bgr, face, src_faces[0], paste_back=True)

        # 4) GFPGAN Restoration
        swapped_rgb = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)
        _, _, restored_rgb = restorer.enhance(
            swapped_rgb,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # Convert restored result back to PIL-friendly RGB
        output_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

        return "✅ Face swap + GFPGAN restoration complete", output_rgb

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# === Gradio Interface ===
demo = gr.Interface(
    fn=swap_faces,
    inputs=[
        gr.Image(label="Source Face", type="pil"),
        gr.Image(label="Target Image", type="pil")
    ],
    outputs=[
        gr.Text(label="Status"),
        gr.Image(label="Swapped & Restored Output")
    ],
    title="Face Swapper with GFPGAN + Auto-Resize",
    description="Uploads too large? We'll auto-resize before face-swap & GFPGAN."
)

demo.launch(share=True)
