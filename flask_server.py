from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

def fmt_prompt(prompt: str) -> str:
    return f"""[Instructions]:\n{prompt}\n\n[Response]:"""

@app.route('/generate', methods=['GET'])
def generate():
    prompt = request.args.get('prompt')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    prompt_input = fmt_prompt(prompt)
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    completion = tokenizer.decode(
        generated_ids[0][input_ids_cutoff:],
        skip_special_tokens=True,
    )

    return jsonify({"response": completion})

if __name__ == "__main__":
    model_name = "ugshanyu/orca-mongol"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
        )
        .to("cuda:0")
        .eval()
    )

    app.run(host='0.0.0.0', port=5000)
