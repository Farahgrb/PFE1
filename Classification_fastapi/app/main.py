# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI, Body
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
Model_Used = "UBC-NLP/MARBERT"
model = BertForSequenceClassification.from_pretrained(Model_Used, num_labels=3)
model_state_dict = torch.load('./app/marbert_80.pth',map_location=torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained(Model_Used)
model.load_state_dict(model_state_dict)
# 2. Create the app object
app = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8001
@app.get('/')
def index():
    return {'message': 'Hello from Classification'}

# 3. Expose the prediction functionality, make a prediction from the passed text
#    and return the predicted label with the confidence
@app.post('/classify')
def detect_hate(text_body = Body()):
    text= text_body.get('text')
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    # Get the predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=1)
    # Assuming you have a list of class labels 
    class_labels = ["Normal", "Abusive", "Discrimination"]
    # Print the predicted label
    predicted_label = class_labels[predicted_labels.item()]
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_probability = probabilities[0][predicted_labels.item()]

    
   # output='{}:{}'.format(text,str(predicted_label))
    output = '{}:{} / Confidence: {:.2f}'.format(text, predicted_label, predicted_probability.item())
    result={"Transcription":text, "label":predicted_label}
    return result

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8001)
    
#uvicorn main:app --reload
