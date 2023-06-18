import random
import json
import torch
from healthchatbot.model import NeuralNet
from healthchatbot.utils import words_container, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("healthchatbot/datafile.json", 'r') as f:
    file_data = json.load(f)

FILE = "healthchatbot/trained_weights.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_size = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_size)
model.eval()

bot_name = "Weller"

def get_response(message):

    sentence = tokenize(message)
    x = words_container(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    out = model(x)
    res, predicted = torch.max(out, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(out, dim=1)
    prob = probs[0][predicted.item()]
    
    if prob.item() > 0.75:

        for obj in file_data['intents']:
            if tag == obj['tag']:
                return random.choice(obj['responses'])
    else:
        return "Sorry! I didn't get that, I am evolving"

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        l = ['Quit', 'quit', 'Quit']
        if sentence in l:
            break

        resp = get_response(sentence)
        print(resp)