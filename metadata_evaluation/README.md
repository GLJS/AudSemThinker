# Video Metadata Extraction Pipeline

This project provides a modular pipeline for extracting various types of metadata from videos, including image analysis, audio analysis, video-audio synchronization, and video captioning.

## Project Structure

The project is organized into four main modules:

### 1. Image Model (`image_model/`)
- Processes video frames for image analysis
- Uses InternVL2 model for image understanding
- Components:
  - `main.py`: Entry point for image processing
  - `dataloader.py`: Handles image data loading and preprocessing
  - `utils/utils.py`: Image-specific utility functions

### 2. Audio Model (`audio_model/`)
- Processes audio streams from videos
- Uses Beats-Conformer-BART for audio captioning
- Components:
  - `main.py`: Entry point for audio processing
  - `dataloader.py`: Handles audio data loading and preprocessing
  - `utils/utils.py`: Audio-specific utility functions

### 4. Video Model (`video_model/`)
- Generates video captions
- Uses LLaVA-Video-7B-Qwen2 for video captioning
- Components:
  - `main.py`: Entry point for video captioning
  - `dataloader.py`: Handles video data loading and preprocessing
  - `utils/utils.py`: Video-specific utility functions

## Models Used

- **Image Analysis**: InternVL2 (5-8B) - Top model from OpenCompass Open VLM Leaderboard
- **Video Captioning**: LLaVA-Video-7B-Qwen2 - Top model from OpenCompass OpenVLM Video Leaderboard
- **Object Detection**: Grounding DINO with Segment Anything
- **Audio Captioning**: Beats-Conformer-BART

## Usage

Each module can be run independently using its respective main.py script. All modules accept similar command-line arguments:

```bash
python <module>/main.py \
    --dataset_csv /path/to/dataset.csv \
    --tar_folder /path/to/video/tars \
    --output_csv /path/to/output.csv \
    --batch_size 4 \
    --num_workers 4
```

### Common Arguments
- `dataset_csv`: Path to CSV file containing video metadata
- `tar_folder`: Path to folder containing tar.gz video files
- `output_csv`: Path to save the output results
- `batch_size`: Batch size for processing (default: 4)
- `num_workers`: Number of worker processes (default: 4)

## Data Format

### Input
- Videos should be provided in tar.gz archives
- Each video file name should follow the format: `{video_id} ({start_time}-{end_time}).{ext}`
- Dataset CSV should contain columns: video_id, start, end, duration

### Output
Each module produces a JSON file containing extracted metadata:
- Image Model: Image descriptions and features
- Audio Model: Audio captions and features
- Video Model: Video captions with confidence scores

## Requirements

- Python 3.8+
- PyTorch 2.0+
- ffmpeg
- Additional dependencies listed in requirements.txt

## References

- Video Captioning: [OpenVLM Video Leaderboard](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard)
- Image Captioning: [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- Object Detection: [Grounding DINO Tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb)


## Installation

```bash
conda create -y -n yt python=3.11
conda activate yt
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 cuda cuda-toolkit -c pytorch -c nvidia
conda install -y -c conda-forge "cmake>=3.18" gcc gxx_linux-64 git
pip install --no-build-isolation flash-attn
pip install -r requirements.txt
python -c "import xgrammar; print(xgrammar)"
```
