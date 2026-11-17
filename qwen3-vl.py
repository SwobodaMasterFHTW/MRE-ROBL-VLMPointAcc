# Qwen3 test script

from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import requests
import torch
import re

checkpoint_dir = "Qwen/Qwen3-VL-4B-Instruct"

# 4-bit quantization for performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=[
        "lm_head",
        "model.vision_backbone.image_vit.patch_embedding",
    ],
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
image = Image.open("carola68-stud-4999947.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)