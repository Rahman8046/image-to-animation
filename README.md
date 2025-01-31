# Anime Image Converter

This application converts regular photos into anime-style artwork. It supports both portrait and landscape/cityscape conversions.

## Features

- Portrait Mode: Convert human portraits into anime-style characters
- Landscape Mode: Convert landscapes and cityscapes into anime-style backgrounds
- User-friendly web interface
- Additional prompt customization for fine-tuning results

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:7860)

3. Choose either Portrait or Landscape mode depending on your image type

4. Upload an image and click the convert button

5. Optionally, add custom prompt details to fine-tune the conversion

## Notes

- The first run will download the necessary models (several GB)
- A CUDA-capable GPU is recommended for faster processing
- The application uses Stable Diffusion with anime-specific models

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- At least 10GB free disk space for models
