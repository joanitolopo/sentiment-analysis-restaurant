import torch
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
from arsitektur import RNN
from utils import tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def predict_sentiment(sentence):
    weights = torch.load("./model/sentiment_model.pt", map_location='cpu')
    config = torch.load("./model/config.pt", map_location='cpu')
    TEXT = torch.load("./data/vocab.pt", map_location='cpu')

    # load arsitektur
    model = RNN(config.INPUT_DIM, config.EMBEDDING_DIM, config.HIDDEN_DIM, config.OUTPUT_DIM, config.N_LAYERS, 
            config.BIDIRECTIONAL, config.DROPOUT, config.PAD_IDX)
    # load model
    model.load_state_dict(weights)
    model = model.to(device)

    # predict
    with torch.no_grad():
        model.eval()
        tokenized = [tok for tok in word_tokenize(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        prediction = torch.sigmoid(model(tensor, length_tensor))

    if prediction.item() > 5:
        result = "Sentiment : Positif"
    elif prediction.item() < 5:
        result = "Sentiment : Negatif"
    
    return {prediction.item():result}


if __name__=="__main__":
    print(predict_sentiment("Restaurantnya jelek."))
