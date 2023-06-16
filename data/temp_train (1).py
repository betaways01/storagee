########## 1. Loading the model
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



########### 2.Load JSON data
# Load JSON data
def load_json_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ids = list(data.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        record_id = self.ids[idx]
        record = self.data[record_id]

        # Access the information within the record
        text = record['QUESTION']
        target = record['final_decision'] #@@@QUESTION: target can be a string? here it's yes, no, or maybe
 
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        #input_ids = tokenizer.encode_plus(text, option, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        
        ##########start#########
        # Get the number of tokens
        result = tokenizer(text)
        #result = tokenizer("The answer to the question given the context is 22 gauge. 22 gauge cannulae are the smallest gauge cannulae available. They are the")
        print("There are {} tokens".format(len(result['input_ids'])))
        num_input_tokens = len(input_ids)
        print("===There are {} tokens".format(num_input_tokens))
        ############end#########

        target_ids = tokenizer.encode(target, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        return {"input_ids": torch.tensor(input_ids), "target_ids": torch.tensor(target_ids)}






########### 3.Load JSON data and create a dataset
#########
#json_file_path = '/Users/ninaz/Desktop/python/datasets/chatdoctor_small.json'
json_file_path = '/workspace/datasets/pubmedqa/pubmedqa-master/data/pqal_fold0/train_set.json'
json_data = load_json_data(json_file_path)
dataset = CustomDataset(json_data, tokenizer, max_length=40)

json_file_path = '/workspace/datasets/pubmedqa/pubmedqa-master/data/pqal_fold1/train_set.json'
json_data = load_json_data(json_file_path)
validationset = CustomDataset(json_data, tokenizer, max_length=40)
###########




############ 4.
# Load JSON data and create a dataset
#json_file_path = '/Users/ninaz/Desktop/python/datasets/chatdoctor_small.json'
json_file_path = '/workspace/datasets/pubmedqa/pubmedqa-master/data/pqal_fold0/train_set.json'
json_data = load_json_data(json_file_path)
dataset = CustomDataset(json_data, tokenizer, max_length=40)

json_file_path = '/workspace/datasets/pubmedqa/pubmedqa-master/data/pqal_fold1/train_set.json'
json_data = load_json_data(json_file_path)
validationset = CustomDataset(json_data, tokenizer, max_length=40)
############



############# 5. double check the encoding is correct
#####
#To fix error: ValueError: Asking to pad but the tokenizer does not have a padding token. 
#Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` 
#or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
#@@@QUESTION: Not sure if this is correct? Or i have to check the model specs?

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenizer.pad_token = tokenizer.eos_token 
# Select a sample
for batch in dataloader:  # Replace 0 with the index of the sample you want to check.
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)
    print(tokenizer.decode(input_ids[0]))
    print(f"Correct answer: {tokenizer.decode(target_ids[0]) }")



############## 6. fine-tuning
tokenizer.pad_token = tokenizer.eos_token 
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    
    for batch in dataloader:
        print("## inside batshing dataloader")
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        outputs = model(input_ids, labels=target_ids)

        print("## got outputs of training")
        loss = outputs.loss
        print(f"## loss:  {loss}") 
        #@@@Question:
        #The problem is that loss is becoming 'nan' after the 1st iteration in this loop. not sure why
        
        total_train_loss += loss.item()
        print(f"## total_train_loss:  {total_train_loss}")
        
        print("## ")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_train_loss = total_train_loss / len(dataloader)
    print(f"## avg_train_loss:  {avg_train_loss}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            print("## inside validation_dataloader dataloader")
            
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            outputs = model(input_ids, labels=target_ids)

            print("## got outputs of validation")
            loss = outputs.loss
            print(f"## val loss:  {loss}")
            
            total_val_loss += loss.item()
            print(f"## total_val_loss:  {loss}")

    avg_val_loss = total_val_loss / len(validation_dataloader)
    print(f"## avg_val_loss:  {avg_val_loss}")

    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

#torch.save(model.state_dict(), f'/workspace/original_biogptlarge/checkpoints/checkpoint_epoch_{epoch}.pt')
#to save checkpoints
#output_dir = "/workspace/trainedmodels/pubmedqabiogptlarge/"
#model.save_pretrained(output_dir)
#tokenizer.save_pretrained(output_dir)

