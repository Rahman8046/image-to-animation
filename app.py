import cv2
import numpy as np
import gradio as gr
from PIL import Image, ImageEnhance
import traceback

def process_image(input_image):
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Convert to PIL Image if needed
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        
        # Convert to RGB mode if not already
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Resize image while maintaining aspect ratio
        width, height = input_image.size
        target_size = 512
        ratio = target_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        input_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(input_image)
        
        # Convert color space
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter for smoothing while preserving edges
        smooth = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        
        # Edge detection using adaptive thresholding
        edges = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 9, 2
        )
        
        # Dilate edges to make them more prominent
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Color quantization
        data = smooth.reshape((-1, 3))
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        K = 8  # Number of colors
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(smooth.shape)
        
        # Convert edges to 3 channels
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Blend quantized image with edges
        result = cv2.addWeighted(quantized, 0.85, edges_colored, 0.15, 0)
        
        # Convert back to RGB for PIL
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for final processing
        output_image = Image.fromarray(result)
        
        # Enhance the image
        enhancer = ImageEnhance.Contrast(output_image)
        output_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Color(output_image)
        output_image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Sharpness(output_image)
        output_image = enhancer.enhance(1.5)
        
        return output_image, "Success"
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # Print to console for debugging
        return None, error_msg

# Create Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Input Image"),
    outputs=[
        gr.Image(type="pil", label="Output Image"),
        gr.Textbox(label="Status")
    ],
    title="Anime Style Image Converter",
    description="Convert your photos into anime-style art using computer vision techniques.",
    examples=[
        ["example1.jpg"],
        ["example2.jpg"]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)
