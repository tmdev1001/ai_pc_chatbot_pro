# chatbot.py
import torch, json
from training_data import simple_embed
from model_train import create_model

# Load the trained model
model = create_model()
model.load_state_dict(torch.load('intent_model.pt'))
model.eval()

def predict_intent(text):
    x = torch.tensor([simple_embed(text)], dtype=torch.float32)
    intent = torch.argmax(model(x)).item()
    return intent

def respond_to_query(text):
    with open('usage_data.json','r') as f:
        usage = json.load(f)
    most_used = max(usage, key=usage.get)
    intent = predict_intent(text)
    print(intent)

    if intent == 0:
        return f"You used {most_used} many times this week."
    else:
        total = sum(usage.values())
        return f"Total active time: {total/60} minutes across {len(usage)} apps this week."

if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        print("Bot: ", respond_to_query(query))