from transformers import pipeline
from transformers import AutoModelForCausalLM

pipe = pipeline("image-text-to-text", model="allenai/Molmo-7B-D-0924", trust_remote_code=True)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
pipe(text=messages)