"""

import os
import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

# SAM3 model load (cpu inference also supported)
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="weights/sam3.pt"))

images_dir = "assets/images"
yolo_ann_dir = "assets/images/yolo_labels"
if not os.path.exists(yolo_ann_dir):
    os.mkdir(yolo_ann_dir)

# Auto annotation
label_to_predict = "bird"
for i, img in enumerate(os.listdir(images_dir)):
    url = os.path.join(images_dir, img)
    image = Image.open(url)  # Image load

    # Run inference with text prompt
    results = processor.set_text_prompt(state=processor.set_image(image),
                                        prompt=label_to_predict)

    # Visualization and auto annotation in YOLO format.
    result_image = draw_box_and_masks(
        cv2.imread(url, cv2.COLOR_RGB2BGR), # PIL -> OpenCV
        results=results,                    # SAM3 predictions
        show_boxes=True,                    # Display bounding boxes on output image
        show_masks=True,                    # Display masks on output image
        mask_alpha=0.4,                     # Adjust mask overlay value, range [0.0 - 1.0]
        show_conf=True,                     # Bool: display object confidence score.
        show_label=True,                    # Bool: display class label.
        line_width=4,                       # Int: Adjust label, box, and mask fontsize.
        label=label_to_predict,             # Str: Bounding box/mask label
        save_yolo=True,                     # Bool: Write annotations in YOLO format.
        filename=os.path.join(yolo_ann_dir, img[:-4]+".txt"),  # Str: Annotation file name.
        class_id=0                          # only useful for bbox and mask color selection.
    )
    print(f"{i+1} Images processed, annotations saved in {yolo_ann_dir}")

"""


import os
import cv2
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

# SAM3 model load (cpu inference also supported)
processor = Sam3Processor(build_sam3_image_model(checkpoint_path="weights/sam3.pt"))

images_dir = "assets/images"

yolo_ann_dir = os.path.join(images_dir, "yolo_labels")
os.makedirs(yolo_ann_dir, exist_ok=True)  # Safer directory creation

# Auto annotation
label_to_predict = "white dog"

# Filter only actual image files (ignores folders like yolo_labels)
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

image_files = [
    f for f in os.listdir(images_dir)
    if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(valid_exts)
]


for i, img_name in enumerate(image_files):
    img_path = os.path.join(images_dir, img_name)
    image = Image.open(img_path).convert("RGB")  # ✅ Ensure consistent RGB format

    # Run inference with text prompt
    results = processor.set_text_prompt(
        state=processor.set_image(image),
        prompt=label_to_predict
    )

    # Proper RGB -> BGR conversion for OpenCV
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Safe annotation path (handles files with multiple dots)
    base_name = os.path.splitext(img_name)[0]
    yolo_path = os.path.join(yolo_ann_dir, f"{base_name}.txt")

    # Visualization and auto annotation in YOLO format
    result_image = draw_box_and_masks(
        image_bgr,
        results=results,
        show_boxes=True,
        show_masks=True,
        mask_alpha=0.4,
        show_conf=True,
        show_label=True,
        line_width=4,
        label=label_to_predict,
        save_yolo=True,
        filename=yolo_path,
        class_id=0
    )
    print(f"[{i+1}/{len(image_files)}] Processed: {img_name} | Annotations saved.")
    results_dir = os.path.join(images_dir, "Results")

    os.makedirs(results_dir, exist_ok=True)
    cv2.imwrite(os.path.join(results_dir, f"result_{base_name}.png"), result_image)

print("✅ Batch processing complete!")