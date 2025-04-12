import os
import onnxruntime
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Load the ONNX model
onnx_model_path = "sar2rgb.onnx"
sess = onnxruntime.InferenceSession(onnx_model_path)

# Function to process the input and make predictions
def predict(input_image):
    # Preprocess the input image
    input_image = input_image.resize((256, 256))  # Adjust size as needed
    input_image = np.array(input_image).transpose(2, 0, 1)  # HWC to CHW
    input_image = input_image.astype(np.float32) / 255.0  # [0,1]
    input_image = (input_image - 0.5) / 0.5               # [-1,1] 
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    
    # Run the model
    inputs = {sess.get_inputs()[0].name: input_image}
    output = sess.run(None, inputs)
    
    # Post-process the output image
    output_image = output[0].squeeze().transpose(1, 2, 0)  # CHW to HWC
    output_image = (output_image + 1) / 2  # [0,1]
    output_image = (output_image * 255).astype(np.uint8)  # Denormalize [0,255]
    output_image_pil = Image.fromarray(output_image)

    # Generate histogram
    fig, ax = plt.subplots()
    ax.hist(output_image.flatten(), bins=256, color='blue', alpha=0.7)
    ax.set_title('Histogram of Pixel Intensities')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    histogram_image = Image.open(buf)
    plt.close(fig)
    
    return output_image_pil, histogram_image

# Specify example images
example_images = [[os.path.join("examples", fname)] for fname in os.listdir("examples")]

# Create Gradio interface
iface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil"), 
    outputs=[gr.Image(type="pil", label="Generated Image"), gr.Image(type="pil", label="Histogram")],
    examples=example_images
)

# Launch the interface
iface.launch()
