import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

device = torch.device("mps" if torch.mps.is_available() else "cpu")

model.eval()
model.to(device)
config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
                          use_cache=True,
                          do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                          return_dict_in_generate=True,
                          eos_token_id=65505,bos_token_id=65504,pad_token_id=65509)


messages = []
messages.append({"role": "system", "content" : "You are a helpful assistant."})
messages.append({"role": "user", "content" : "What is 123 times 246?"})
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(chat_input)
input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

outputs = model.generate(input_ids, config, num_steps=16, tokenizer=tokenizer)

print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
