from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("./trained_model")
tokenizer = BartTokenizer.from_pretrained("./trained_model")

def generate_follow_up(question, answer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_text = question + "<SEP>" + answer + "<QUS>"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

question = "ELI5 Do animals tan?"
answer = "Animals can get sunburned like we do, pigs for example root in mud to cool off and because mud is a form of sun protection. Animals with fur cannot get sunburned where and when the fur covers them but can get burned on their lips and eyelids."
follow_up = generate_follow_up(question, answer)
print(follow_up)

question = "ELI5 What is GERD?"
answer = "Gerd (GastroEsophageal Reflux Disease) is a problem with the sphincter that leads into the stomach. If the sphincter doesn't close properly, stomach acids can be pumped back up the esophagus. The acid is an irritant, the mucus is what the delicate tissues use to defend themselves.  Reflux is a common cause of heartburn. Constant reflux can cause cancer of the sphincter."
follow_up = generate_follow_up(question, answer)
print(follow_up)

question = "eli5 Why didn\u2019t the dwarves fight in the War of the Ring? Wouldn\u2019t the outcome affect them too?"
answer = "They did, the movies skip a lot of side details about other conflicts happening in the world that you get in the books. Though.... they're quick notes and easy to miss/forget even in text."
follow_up = generate_follow_up(question, answer)
print(follow_up)