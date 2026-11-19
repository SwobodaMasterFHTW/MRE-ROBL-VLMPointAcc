# Molmo2 test script

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
import torch
import re

checkpoint_dir = "allenai/Molmo2-4B"  # or path to a converted HF checkpoint

# 4-bit quantization for performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["model.vision_backbone.image_vit.patch_embedding"],
)

# Setup model
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint_dir,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    quantization_config=quantization_config,
    device_map={"": 0},
)

processor = AutoProcessor.from_pretrained(
    checkpoint_dir,
    trust_remote_code=True,
    padding_side="left",
)

# Load image
image = Image.open("reference/carola68-stud-4999947.jpg").convert("RGB")

image_messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Point to the exact geometric centre of the green button."
                    "Return the point coordinates, nothing else."
                )
            },
            {"type": "image", "image": image},
        ]
    }
]

# Define input
inputs = processor.apply_chat_template(
    image_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    padding=True,
)

inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model.generate(**inputs, max_new_tokens=200)

generated_tokens = output[0, inputs["input_ids"].size(1):]
result = processor.decode(generated_tokens, skip_special_tokens=False)

print("Model output:", result)

# Extract point coordinates
match = re.search(
    r'<points\s+coords="([^"]+)"',
    result
)

if match:
    coords = [float(x) for x in match.group(1).split()]

    # Molmo point format: first value is the point label/index,
    # followed by X and Y coordinates.
    if len(coords) >= 3:
        normalized_x, normalized_y = coords[-2:]

        # Convert normalized coordinates to image pixels
        pixel_x = round(normalized_x / 1000 * image.width)
        pixel_y = round(normalized_y / 1000 * image.height)

        print(f"Button centre: ({pixel_x}, {pixel_y})")
    else:
        print("Unexpected coordinate format:", coords)
else:
    print("No point coordinates found in model output.")
