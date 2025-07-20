import cv2
import os
import gradio as gr
import numpy as np
import insightface
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from numpy.linalg import norm
from numpy import dot

# === Init InsightFace (GPU + swapper)
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = get_model('inswapper_128', download=False)

# === Helper: cosine similarity
def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# === Main face swap video function
def swap_faces_in_video(source_img, target_video, downscale=True, max_frames=300, similarity_threshold=0.6):
    # Save source image locally
    source_path = "temp_source.jpg"
    output_path = "swapped_output.mp4"

    source_img.save(source_path)
    target_path = target_video.name  # ✅ use Gradio's auto-saved file path

    # Load source face
    src_img = cv2.imread(source_path)
    src_faces = app.get(src_img)
    if not src_faces:
        return "❌ No face detected in source image.", None
    src_face = src_faces[0]

    # Open video
    cap = cv2.VideoCapture(target_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if downscale:
        width //= 2
        height //= 2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in tqdm(range(min(total_frames, max_frames)), desc="Swapping..."):
        ret, frame = cap.read()
        if not ret:
            break

        if downscale:
            frame = cv2.resize(frame, (width, height))

        dst_faces = app.get(frame)
        for dst_face in dst_faces:
            similarity = cosine_similarity(dst_face.normed_embedding, src_face.normed_embedding)
            if similarity > similarity_threshold:
                frame = swapper.get(frame, dst_face, src_face, paste_back=True)

        out.write(frame)

    cap.release()
    out.release()

    return "✅ Swap complete!", output_path

# === Gradio UI ===
with gr.Blocks(title="Face Swapper with Tracking & Optimization") as demo:
    gr.Markdown("# 🎥 InsightFace Video Face Swapper")
    gr.Markdown("Upload a source face and a video. We'll swap matching faces using tracking + speed boosts.")

    with gr.Row():
        source_input = gr.Image(label="Source Face (Image)", type="pil")
        video_input = gr.File(label="Target Video (.mp4)", file_types=[".mp4"])

    with gr.Accordion("⚙️ Advanced Options", open=False):
        downscale_opt = gr.Checkbox(value=True, label="Downscale Video for Faster Processing")
        frame_limit = gr.Slider(50, 1000, value=300, step=10, label="Max Frames to Process")
        sim_thresh = gr.Slider(0.4, 0.9, value=0.6, step=0.01, label="Tracking Similarity Threshold")

    run_button = gr.Button("🚀 Run Face Swap")
    status_output = gr.Textbox(label="Status")
    video_output = gr.File(label="Download Swapped Video")

    run_button.click(
        fn=swap_faces_in_video,
        inputs=[source_input, video_input, downscale_opt, frame_limit, sim_thresh],
        outputs=[status_output, video_output]
    )

demo.launch(share=True)
