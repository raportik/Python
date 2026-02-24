from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch

MODEL_PATH = "zai-org/GLM-OCR"
PROMPT = "Text Recognition:"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

image = Image.open("captcha.png").convert("RGBA")
background = Image.new("RGBA", image.size, (255, 255, 255))
combined = Image.alpha_composite(background, image).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": combined},
            {"type": "text", "text": PROMPT},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to(device)
inputs.pop("token_type_ids", None)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)

gen_ids = out[:, inputs["input_ids"].shape[1] :]
text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
print(text)
