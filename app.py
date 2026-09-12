import os
import numpy as np
import gradio as gr

from PIL import Image, ImageChops, ImageFilter


# ==========================================
# 1. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):
    """
    Resize and convert the image into a standard format.
    """

    image = image.convert("RGB")
    image = image.resize((224, 224))

    return image


# ==========================================
# 2. EXTRACT VISUAL FEATURES
# ==========================================

def extract_features(image):
    """
    Extract simple visual features from the image.
    """

    image = preprocess_image(image)

    # Convert image to grayscale
    gray = image.convert("L")

    # Convert pixels into NumPy array
    pixels = np.array(gray)

    # Average brightness
    brightness = np.mean(pixels)

    # Standard deviation shows image variation
    variation = np.std(pixels)

    # Detect edges
    edges = gray.filter(ImageFilter.FIND_EDGES)

    edge_array = np.array(edges)

    # Average edge strength
    edge_strength = np.mean(edge_array)

    return brightness, variation, edge_strength


# ==========================================
# 3. DEFECT DETECTION
# ==========================================

def detect_defect(image):

    if image is None:
        return "Please upload a product image.", 0, "No image provided."

    brightness, variation, edge_strength = extract_features(image)

    # --------------------------------------
    # Calculate defect score
    # --------------------------------------

    defect_score = 0

    # Very high variation can indicate
    # scratches, cracks or irregular surfaces
    if variation > 65:
        defect_score += 40

    elif variation > 45:
        defect_score += 20

    # Strong edges can indicate irregularities
    if edge_strength > 35:
        defect_score += 40

    elif edge_strength > 25:
        defect_score += 20

    # Extremely dark or bright images
    # may indicate inspection problems
    if brightness < 40 or brightness > 220:
        defect_score += 20

    # --------------------------------------
    # Final classification
    # --------------------------------------

    if defect_score >= 50:

        result = "🔴 DEFECT DETECTED"

        explanation = (
            "The image contains strong visual irregularities. "
            "The system detected high variation or edge activity."
        )

    else:

        result = "🟢 NO DEFECT DETECTED"

        explanation = (
            "The image appears visually consistent "
            "with a normal product surface."
        )

    return result, defect_score, explanation


# ==========================================
# 4. IMAGE PREPROCESSING PREVIEW
# ==========================================

def preprocess_preview(image):

    if image is None:
        return None

    return preprocess_image(image)


# ==========================================
# 5. GRADIO USER INTERFACE
# ==========================================

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # 🔍 AI-Powered Visual Defect Detection

        Upload a product image and the system will
        analyze its visual characteristics to detect
        possible defects.
        """
    )

    gr.Markdown(
        """
        **Detection pipeline:**

        Image → Preprocessing → Feature Extraction
        → Defect Score → Inspection Result
        """
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Upload Product Image"
        )

        image_output = gr.Image(
            type="pil",
            label="Preprocessed Image"
        )

    preprocess_button = gr.Button(
        "🖼️ Preprocess Image"
    )

    preprocess_button.click(
        fn=preprocess_preview,
        inputs=image_input,
        outputs=image_output
    )

    detect_button = gr.Button(
        "🔍 Detect Defect"
    )

    result = gr.Textbox(
        label="Inspection Result"
    )

    score = gr.Number(
        label="Defect Score"
    )

    explanation = gr.Textbox(
        label="Explainable Insight"
    )

    detect_button.click(
        fn=detect_defect,
        inputs=image_input,
        outputs=[
            result,
            score,
            explanation
        ]
    )

    gr.Markdown(
        """
        ### 💡 How it works

        The system preprocesses the image, extracts
        brightness, variation and edge information,
        and uses these visual features to estimate
        whether an image contains possible defects.

        This lightweight version is designed for
        resource-efficient cloud deployment.
        """
    )


# ==========================================
# 6. RENDER SERVER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
