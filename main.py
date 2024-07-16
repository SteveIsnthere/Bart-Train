import json
from datasets import Dataset
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer
import torch

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return Dataset.from_list(data)

train_data = load_data('train.json') 
valid_data = load_data('valid.json')
test_data = load_data('test.json')

# Load model and tokenizer
model_name = "facebook/bart-large"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Preprocess data
def preprocess_function(examples):
    inputs = [q + " " + a for q, a in zip(examples['question'], examples['answer'])]
    targets = examples['follow-up']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=1024, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_valid_data = valid_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    eval_dataset=tokenized_valid_data,
)

trainer.train()

# Evaluate the model
results = trainer.evaluate(tokenized_test_data)

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")


def generate_follow_up(question, answer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")

    model.to(device)
    input_text = question + "<SEP>" + answer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)