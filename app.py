from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load Qwen3-4B customer support model
model_name = "ragib01/Qwen3-4B-customer-support"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # Automatically place model on GPU
    torch_dtype=torch.float16
)

def chat(user_input):
    input_text = f"User: {user_input}\nAssistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/chat", methods=["POST"])
def chat_api():
    user_message = request.json.get("message")
    response = chat(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
