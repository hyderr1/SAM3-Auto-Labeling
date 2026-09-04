

#original code
"""
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks


label_to_predict = "white dog"  # this will be used as prompt for inference.

url = "assets/dog.jpg"
image = Image.open(url)  # Image load


# SAM3 model load
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="sam3.pt"))

# Run inference with text prompt
results = processor.set_text_prompt(state=processor.set_image(image),
                                    prompt=label_to_predict)

# Visualization
result_image = draw_box_and_masks(cv2.imread(url, cv2.COLOR_RGB2BGR),  # PIL -> OpenCV
                                  results=results,
                                  show_boxes=True,
                                  show_masks=True,
                                  line_width=4,
                                  label=label_to_predict)

cv2.imwrite("sam3_results.png", result_image)  # Save (optional)

"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

SCRIPT_DIR = Path(__file__).parent
IMAGE_PATH = SCRIPT_DIR / "assets" / "dogs.jpg"
CHECKPOINT_PATH = SCRIPT_DIR / "weights" / "sam3.pt"  # Adjust if your .pt is elsewhere


# Verify image exists
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"❌ Image not found at: {IMAGE_PATH.absolute()}\n"
        f"💡 Create an 'assets' folder in the script directory and place 'dog.jpg' there."
    )

label_to_predict = "white dog"

# 2️⃣ Load image with PIL (ensures RGB)
image = Image.open(IMAGE_PATH).convert("RGB")

# SAM3 model load
print("⏳ Loading SAM3 model...")
processor = Sam3Processor(build_sam3_image_model(checkpoint_path=str(CHECKPOINT_PATH)))

# Run inference with text prompt
print(f"🔍 Predicting: '{label_to_predict}'")
results = processor.set_text_prompt(
    state=processor.set_image(image),
    prompt=label_to_predict
)

# 3️⃣ Convert PIL (RGB) → OpenCV (BGR) for visualization
image_np_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Visualization
result_image = draw_box_and_masks(
    image_np_bgr,
    results=results,
    show_boxes=True,
    show_masks=True,
    line_width=4,
    label=label_to_predict
)

# Create 'Results' folder next to the script if it doesn't exist
RESULTS_DIR = SCRIPT_DIR / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save output inside the Results folder
output_path = RESULTS_DIR / "sam3_results.png"
cv2.imwrite(str(output_path), result_image)
print(f"✅ Results saved to: {output_path}")
