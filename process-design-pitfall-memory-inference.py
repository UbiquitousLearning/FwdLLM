import torch
# distilbert-base-uncased, albert-base-v2, bert-large uncased
from transformers import BertTokenizer, DistilBertTokenizer,AlbertTokenizer, BertForSequenceClassification, DistilBertForSequenceClassification, AlbertForSequenceClassification
from tqdm import tqdm
# print process id
import os
from torch.nn import CrossEntropyLoss
print('Process ID:', os.getpid())

model = 'Bert-large'
device = 'cpu'
method = "adapter"
train = True
if model == 'DistilBert-base':
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
elif model == 'Bert-base':
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name).to(device)
elif model == 'Bert-large':
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name).to(device)
elif model == 'Albert-base':
    model_name = 'albert-base-v2'
    tokenizer = AlbertTokenizer.from_pretrained(model_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name).to(device)
else:
    raise ValueError

# sentences = ["how are you?",
#     "I am doing great!",
#     "What's your name?",
#     "Nice to meet you.",
#     "How can I help you?",
#     "This is a sample sentence.",
#     "BERT is awesome.",
#     "Let's do some inference.",
#     "Transformers library is fantastic.",
#     "I love working with deep learning models.",
#     "Natural Language Processing is fascinating.",
#     "Machine learning is revolutionizing the world.",
#     "Python is a versatile programming language.",
#     "OpenAI's GPT-3 is a remarkable model.",
#     "I enjoy chatting with ChatGPT.",
#     "Artificial Intelligence has great potential.",
#     "Deep learning is a subfield of machine learning.",
#     "I am excited to see future advancements.",
#     "The possibilities are endless with AI.",
#     "I wonder what AI will achieve in the future.",
#     "ChatGPT is an AI language model developed by OpenAI.",
#     "I'm impressed with the capabilities of ChatGPT.",
#     "AI can enhance various industries.",
#     "I'm looking forward to AI-powered applications.",
#     "It's fascinating to witness AI progress.",
#     "I'm curious about AI ethics and fairness.",
#     "The AI field is constantly evolving.",
#     "AI has the potential to solve complex problems.",
#     "I'm excited to explore AI further.",
#     "AI can help us make better decisions.",
#     "I'm amazed by the advancements in AI research.",
#     "The future of AI is promising."
# ]*200

sentences = ["future "*256]*200

encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']

batch_size = 8
num_samples = len(input_ids)

# Adjust the number of samples to be divisible by the batch size
num_batches = (num_samples + batch_size - 1) // batch_size
num_samples_adjusted = num_batches * batch_size

# Pad the inputs to match the adjusted number of samples
input_ids = torch.cat([input_ids, input_ids[:(num_samples_adjusted - num_samples)]])
attention_mask = torch.cat([attention_mask, attention_mask[:(num_samples_adjusted - num_samples)]])

# Reshape the inputs to match the batch size
input_ids = input_ids.reshape(num_batches, batch_size, -1)
attention_mask = attention_mask.reshape(num_batches, batch_size, -1)

outputs = []

if method == "adapter":
    model.add_adapter("aaa")
    model.train_adapter("aaa")

if method == "bitfit":
    for n,p in model.named_parameters():
        if not("bias" in n or "classifier" in n):
            p.requires_grad = False

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(model))
if train:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
else:
    model.eval()

# Iterate over each batch
for batch_input_ids, batch_attention_mask in tqdm(zip(input_ids, attention_mask), total=num_batches, desc="Inference progress"):


    # Move the batch to the GPU if available
    batch_input_ids = batch_input_ids.to(device)
    batch_attention_mask = batch_attention_mask.to(device)

    if train:
        # Perform forward pass
        batch_outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        logits = batch_outputs[0]
        labels = torch.ones((batch_size),dtype=torch.int64).to(device)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        with torch.no_grad():
            batch_outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
    
