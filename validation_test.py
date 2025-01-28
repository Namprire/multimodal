import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from torch.amp import autocast
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

# Read the CSV file containing the questions and options
csv_file = 'Validation/validation_without_answers.csv'
df = pd.read_csv(csv_file)

# Get the list of image files from the validation folder
image_folder = 'Validation/images'
image_files = os.listdir(image_folder)

# Prepare the output CSV file
output_csv_file = 'test_validation.csv'
output_data = []

# Loop through each image file in the validation folder
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Get the corresponding row from the CSV file based on the image file name
    row = df[df['file_name'] == image_file].iloc[0]
    question = row['question']
    options = [row['option1'], row['option2'], row['option3'], row['option4']]

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
    with autocast('cuda'):
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

    # Append the result to the output data
    output_data.append([image_file, model_answer])

    del inputs
    del generated_ids
    del generated_ids_trimmed
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Write the output data to the CSV file
output_df = pd.DataFrame(output_data, columns=['file_name', 'answer'])
output_df.to_csv(output_csv_file, index=False)

print(f"Validation results saved to {output_csv_file}")