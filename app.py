from flask import Flask, render_template, request
from model import model, vocab_src, vocab_tgt, idx2word_tgt, tokenize, numericalize
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        src_sentence = request.form["src_sentence"]
        tokens = numericalize(src_sentence, vocab_src)
        src_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

        hidden = model.encoder(src_tensor)
        input_token = torch.tensor([vocab_tgt["<bos>"]], device=device)

        translated_tokens = []
        for _ in range(20):  # max 20 tokens
            output, hidden = model.decoder(input_token, hidden)
            next_token = output.argmax(1).item()
            if idx2word_tgt[next_token] == "<eos>":
                break
            translated_tokens.append(idx2word_tgt[next_token])
            input_token = torch.tensor([next_token], device=device)

        prediction = " ".join(translated_tokens)

    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

