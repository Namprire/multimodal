#this code only reads the first image file from the image folder 
# and the corresponding row from the CSV file based on the image file name.
# and gives us the answer and the reasoning 
#only one image file is read and the corresponding row is taken from the CSV file

import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from torch.cuda.amp import autocast
import os

# Clear the CUDA cache to free up memory
torch.cuda.empty_cache()

# Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Load the default processor for the model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Read the CSV file containing the questions and answers
csv_file = 'Train/train_with_answers.csv'
df = pd.read_csv(csv_file)

# Get the first image file name from the image folder
image_folder = 'Train/images'
image_files = os.listdir(image_folder)
first_image_file = image_files[0]
image_path = os.path.join(image_folder, first_image_file)

# Get the corresponding row from the CSV file based on the image file name
row = df[df['file_name'] == first_image_file].iloc[0]
question = row['question']
options = [row['option1'], row['option2'], row['option3'], row['option4']]
actual_answer = row['answer']

# Prepare the options text for the prompt
options_text = "\n".join([f"{chr(97 + i)}) {option}" for i, option in enumerate(options)])
prompt = f"Question: {question}\nOptions:\n{options_text}\nPlease choose the correct option (a, b, c, or d) and explain why you chose this answer:"

# Prepare the message for the model, including the image and the prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ],
    }
]

# Apply the chat template to the messages and process the vision information
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

# Prepare the inputs for the model
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generate the output from the model
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Find the model's answer among the options
model_answer = None
for i, option in enumerate(options, 1):
    if option in output_text[0]:
        model_answer = i
        break

# Print the model's answer and compare it with the actual answer
print(f"Image file: {first_image_file}")
print(f"Question: {question}")
print(f"Output text: {output_text}")
print(f"Options: {options}")
print(f"Model's answer: {model_answer}")
print(f"Actual answer: {actual_answer}")
print(f"Is the model's answer correct? {'Yes' if model_answer == actual_answer else 'No'}")
print(f"Model's explanation: {output_text}")