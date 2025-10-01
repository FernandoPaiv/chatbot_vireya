import torch
import json
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import tiktoken
from generate_copy import generate
from model_copy import GPTModel
from deep_translator import GoogleTranslator
import os
from dotenv import load_dotenv
load_dotenv()
tokenizer = tiktoken.get_encoding("gpt2")

model_path = "GPT2-ETAs_model/GPT2-ETAs_weights.pth"
config_path = 'GPT2-ETAs_model/GPT2-ETAs_config.json'
with open(config_path, "r") as f:
    config = json.load(f)
config = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'n_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True
}

model = GPTModel(config)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
app = FastAPI()

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != os.getenv('Token'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
        )

class Query(BaseModel):
    pergunta: str

@app.post("/chat")
def chat(query: Query, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    pergunta_en = GoogleTranslator(source='pt', target='en').translate(query.pergunta)
    prompt = f"Pergunta:{pergunta_en}\nResposta:"
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated_idx = generate(
        model, idx, 
        max_new_tokens=100, 
        context_size=config["context_length"], 
        temperature=0.7, 
        top_k=50
    )
    generated_text = tokenizer.decode(generated_idx[0].tolist())

    resposta = str(generated_text).split("Resposta:")[1]
    resposta_pt = GoogleTranslator(source='en', target='pt').translate(resposta)

    return resposta_pt