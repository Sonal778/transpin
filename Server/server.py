import random

import uvicorn
from fastapi import FastAPI, Request

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

import tensorflow_hub as hub

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
        if paraphrase.lower() != sentence.lower() and paraphrase not in temp:
            temp.append(paraphrase)

    if len(temp) < 1:
        temp1 = temp
        sentence = temp1[random.randint(0, len(temp1)-1)]

        model_output = run_model(sentence, top_k, top_p, tokenizer, model)
        temp = preprocess_output(model_output, tokenizer, temp, sentence, model)
    return temp


@app.post("/spin")
async def spinner(request: Request):
    params = await request.json()
    sentence = params['sentence']
    top_k1 = params['top_k']
    top_p1 = params['top_p']
    print(sentence)

    global input_sentence
    global top_k
    global top_p
    input_sentence = sentence
    top_p = top_p1
    top_k = top_k1

    model = T5ForConditionalGeneration.from_pretrained('Vamsi/T5_Paraphrase_Paws')
    tokenizer = T5Tokenizer.from_pretrained('Vamsi/T5_Paraphrase_Paws')

    model_output = run_model(sentence, top_k, top_p, tokenizer, model)

    paraphrases = []
    temp = []

    temp = preprocess_output(model_output, tokenizer, temp, sentence, model)

    global output_cache
    output_cache = temp

    for i, line in enumerate(temp):
        paraphrases.append(f"{i + 1}. {line}")

    print({"data": paraphrases})
    return {"data": paraphrases}


@app.post("/hello")
def helo():
    return {"data": "Hello World"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
