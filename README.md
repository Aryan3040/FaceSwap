# FaceSwap - AI-Powered Face Swapping Tool

A sophisticated face swapping application that combines InsightFace for face detection and swapping with GFPGAN for high-quality face restoration and upscaling.

## Features

- **Advanced Face Detection**: Uses InsightFace for precise face detection and alignment
- **High-Quality Face Swapping**: Seamless face replacement with realistic results
- **Face Restoration**: GFPGAN integration for enhanced face quality and upscaling
- **Auto-Resize**: Automatically resizes large images for optimal processing
- **Gradio Interface**: User-friendly web interface for easy interaction
- **GPU Acceleration**: CUDA support for faster processing

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for optimal performance

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd faceswap
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**
   ```bash
   # Download GFPGAN model (333MB)
   wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
   
   # Or download manually from:
   # https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
   ```

4. **Run the application:**
   ```bash
   python swapup.py
   ```

The Gradio interface will open at `http://localhost:7860`

## How It Works

### Processing Pipeline

1. **Image Preprocessing**: Auto-resize large images to prevent memory issues
2. **Face Detection**: InsightFace detects faces in both source and target images
3. **Face Swapping**: Replace target faces with source face using InsightFace swapper
4. **Face Restoration**: Apply GFPGAN to enhance face quality and upscale
5. **Output Generation**: Return the final swapped and restored image

### Technical Details

- **InsightFace**: State-of-the-art face recognition and swapping
- **GFPGAN**: Face restoration and upscaling using generative adversarial networks
- **Auto-Resize**: Maintains aspect ratio while limiting maximum dimension to 1200px
- **CUDA Optimization**: GPU-accelerated processing for faster results

## File Structure

```
faceswap/
├── swapup.py           # Main Gradio interface
├── vidswap.py          # Video face swapping utility
├── swap.py             # Basic face swapping script
├── GFPGANv1.4.pth      # GFPGAN model (download separately)
├── gfpgan/             # GFPGAN library files
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Usage

### Web Interface (Recommended)

1. **Start the application:**
   ```bash
   python swapup.py
   ```

2. **Upload images:**
   - **Source Face**: The face you want to swap from
   - **Target Image**: The image where you want to place the face

3. **Get results:**
   - View the status message for processing feedback
   - Download the swapped and restored image

### Command Line Usage

```bash
# Basic face swapping
python swap.py

# Video face swapping
python vidswap.py
```

## Configuration

### Model Settings

- **InsightFace Model**: `buffalo_l` (default, good balance of speed/accuracy)
- **GFPGAN Upscale**: 2x upscaling (configurable)
- **Max Image Size**: 1200px (prevents memory issues)

### Customization

```python
# In swapup.py, modify these constants:
MAX_DIM = 1200  # Maximum image dimension
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
restorer = GFPGANer(
    model_path="./GFPGANv1.4.pth",
    upscale=2,  # Change upscale factor
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)
```

## Troubleshooting

### Common Issues

1. **"No face detected"**: 
   - Ensure images contain clear, front-facing faces
   - Try different images with better lighting
   - Check if faces are not too small in the image

2. **CUDA out of memory**:
   - Reduce `MAX_DIM` in the code
   - Use smaller input images
   - Close other GPU applications

3. **Model file missing**:
   - Download `GFPGANv1.4.pth` from the official repository
   - Place it in the project root directory

4. **Poor quality results**:
   - Use high-resolution source images
   - Ensure good lighting in both images
   - Try different face angles

### Performance Tips

- **GPU Memory**: Ensure at least 8GB VRAM for optimal performance
- **Image Size**: Larger images = better quality but slower processing
- **Batch Processing**: Process multiple images sequentially for better memory management

## Advanced Usage

### Video Face Swapping

```bash
python vidswap.py
```

### Custom Face Swapping

```python
from swapup import swap_faces
from PIL import Image

# Load images
source_img = Image.open("source.jpg")
target_img = Image.open("target.jpg")

# Perform face swap
status, result = swap_faces(source_img, target_img)
if result is not None:
    result.save("output.jpg")
```

### Batch Processing

```python
import os
from swapup import swap_faces
from PIL import Image

source_img = Image.open("source.jpg")
target_dir = "target_images/"

for filename in os.listdir(target_dir):
    if filename.endswith(('.jpg', '.png')):
        target_img = Image.open(os.path.join(target_dir, filename))
        status, result = swap_faces(source_img, target_img)
        if result is not None:
            result.save(f"output_{filename}")
```

## Deployment

### Local Development
```bash
python swapup.py
```

### Production Deployment
```bash
# Using gunicorn with Gradio
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:7860 swapup:demo.app

# Or use Gradio's built-in server
python swapup.py --server.port 7860 --server.address 0.0.0.0
```

## Privacy and Ethics

### Important Considerations

- **Consent**: Always obtain consent before swapping faces
- **Privacy**: Respect individuals' privacy rights
- **Misuse**: Do not use for deception or harmful purposes
- **Legal Compliance**: Follow local laws regarding deepfakes and face manipulation

### Best Practices

- Use only images you have permission to modify
- Clearly label AI-generated content
- Respect copyright and intellectual property rights
- Consider the potential impact on individuals

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) for face detection and swapping
- [GFPGAN](https://github.com/TencentARC/GFPGAN) for face restoration
- [Gradio](https://gradio.app/) for the web interface
- [OpenCV](https://opencv.org/) for image processing

## Future Enhancements

- [ ] Video processing improvements
- [ ] Multiple face swapping
- [ ] Real-time face swapping
- [ ] Better face alignment
- [ ] Style transfer options
- [ ] Mobile app integration 