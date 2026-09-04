import cv2
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualize.utils import draw_box_and_masks

# === Settings ===
label_to_predict = "dog"
input_video = "path/to/video.mp4"
output_video = "output_sam3.avi"
model_path = "sam3.pt"

# === LOAD MODEL ===
print("[INFO] Loading SAM3 model...")
processor = Sam3Processor(build_sam3_image_model(checkpoint_path=model_path))

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error opening video file")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_count = 0

# === PROCESS VIDEO ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"[INFO] Processing frame {frame_count}")

    # OpenCV (BGR) -> PIL (RGB)
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Run inference
    state = processor.set_image(image_pil)
    results = processor.set_text_prompt(state=state, prompt=label_to_predict)

    # Draw bbox + mask
    output_frame = draw_box_and_masks(frame, results=results, show_boxes=True,
                                      show_masks=True, line_width=3, label=label_to_predict)

    writer.write(output_frame)  # Write processed frame

# === CLEANUP ===
cap.release()
writer.release()
cv2.destroyAllWindows()