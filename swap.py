import cv2
import insightface
import numpy as np
import gradio as gr
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# === Setup ===
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# === Load Swapper (from HuggingFace or local cache)
swapper = get_model('inswapper_128', download=False)

# === Gradio Function ===
def swap_faces(source_img, target_img):
    try:
        # Convert to OpenCV format
        src_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        dst_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        # Detect faces
        src_faces = app.get(src_img)
        dst_faces = app.get(dst_img)

        if not src_faces:
            return "❌ No face detected in source image", None
        if not dst_faces:
            return "❌ No face detected in target image", None

        # Swap faces
        for face in dst_faces:
            dst_img = swapper.get(dst_img, face, src_faces[0], paste_back=True)

        # Convert back to PIL format for Gradio
        output_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)

        return "✅ Swap complete", output_img

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# === Gradio UI ===
demo = gr.Interface(
    fn=swap_faces,
    inputs=[
        gr.Image(label="Source Face", type="pil"),
        gr.Image(label="Target Image", type="pil")
    ],
    outputs=[
        gr.Text(label="Status"),
        gr.Image(label="Swapped Output")
    ],
    title="Face Swapper with InsightFace",
    description="Upload a source face and a target image. The source face will be swapped onto all faces in the target."
)

demo.launch(share=True)
