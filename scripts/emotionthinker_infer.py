import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

processor = Qwen2_5OmniProcessor.from_pretrained('ddwang2000/EmotionThinker')

model = Qwen2_5OmniForConditionalGeneration.from_pretrained('ddwang2000/EmotionThinker',torch_dtype="auto", device_map="auto")

print("✅ Model loaded successfully")

audio_path="./example/angry.wav" #your audio path
prompt="<audio>What is the emotion expressed in this audio clip? Please choose one from the following options: neutral, happy, sad, angry, contempt or disgust, confused, whisper, surprise, fear."

messages = [
    {"role": "system", "content": [
        {"type": "text", "text": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."} 
    ], },
    {"role": "user", "content": [
        {"type": "audio", "audio": audio_path},
        {"type": "text", "text": prompt},
    ]
     },
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
inputs = inputs.to(model.device).to(model.dtype)

with torch.no_grad():
    text_ids = model.generate(
        **inputs,
        return_audio=False,
        max_new_tokens=2048
    )[:, inputs.input_ids.size(1):]


text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)
