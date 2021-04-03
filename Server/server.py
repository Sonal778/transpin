import random

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


app = FastAPI()

output_cache = []
input_sentence = ""


def run_model(sentence, top_k, top_p, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    text = "paraphrase: " + sentence + " </s>"

    max_len = 256

    encoding = tokenizer.encode_plus(text, padding='longest', return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    
    outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=max_len,
            top_k=top_k,
            top_p=top_p,
            early_stopping=False,
            # temperature=decoding_params["temperature"],
            num_return_sequences=1  # Number of sentences to return
        )

    return outputs



def preprocess_output(model_output, tokenizer, temp, sentence, model):
    for line in model_output:
        paraphrase = tokenizer.decode(line, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return paraphrase

class Item(BaseModel):
    sentence: str
    originality: int
    accuracy: float


@app.post("/spin")
async def spinner(item: Item):
    sentence1 = item.sentence
    top_k1 = item.originality
    top_p1 = item.accuracy

    global input_sentence
    global top_k
    global top_p
    input_sentence = sentence1
    top_k = top_k1
    top_p = top_p1
    
    model = T5ForConditionalGeneration.from_pretrained('/Server/Model/')
    tokenizer = T5Tokenizer.from_pretrained('/Server/Token/')

    model_output = await run_model(input_sentence, top_k, top_p, tokenizer, model)

    paraphrases = []
    temp = []

    temp = await preprocess_output(model_output, tokenizer, temp, input_sentence, model)

    print({"data": temp})
    return {"data": temp}


@app.get("/hello")
def helo():
    return {"data": "Hello World"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app,host='0.0.0.0',port=port)
