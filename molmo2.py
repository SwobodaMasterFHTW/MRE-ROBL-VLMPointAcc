from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
import torch

checkpoint_dir = "allenai/Molmo2-4B"  # or path to a converted HF checkpoint

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_skip_modules=["model.vision_backbone.image_vit.patch_embedding"],
)

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

image_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image", "image": Image.open("wikiimages-mars-67522_1920.jpg")},
        ]
    }
]

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
print(processor.decode(generated_tokens, skip_special_tokens=True))