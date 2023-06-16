from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from torch.optim import AdamW
import json
#from datasets import Dataset
import torch
from torch.utils.data import  DataLoader, Dataset
#from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline

custom_pretrained_model_path = "/workspace/falcon40b/falcon-40b"

model = AutoModelForCausalLM.from_pretrained(custom_pretrained_model_path, trust_remote_code=True, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(custom_pretrained_model_path)

print("### loaded the model")



# Start interaction loop
# for as long as you like, until you type quit
tokenizer.add_special_tokens({'pad_token': '\n'})
while True:
    # Ask user input
    input_text = input("You: ")

    # Quit if the user types 'quit'
    if input_text.lower() == 'quit':
        break

    # Encode input text
    textToSend = f"Question: {input_text} . \n Answer: "
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate response from model
    output_ids = model.generate(input_ids)

    # Decode output IDs to get model's answer
    output_text = tokenizer.decode(output_ids[0], num_return_sequences=1, max_length="128")
    print(output_text)
    # Print the model's answer
    print(f"Falcon-40b: {output_text}")


#Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?
#Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death? \n Options: A) Yes \n B) No C) Maybe \n Answer: #
#Is adjustment for reporting heterogeneity necessary in sleep disorders?
#Do mutations causing low HDL-C promote increased carotid intima-media thickness?
#Question: Is adjustment for reporting heterogeneity necessary in sleep disorders? \n Options: A) Yes \n B) No C) Maybe \n Answer:
#Question: Can tailored interventions increase mammography use among HMO women? \n Options: \n A) Yes \n B) No \n C) Maybe \n Answer:

