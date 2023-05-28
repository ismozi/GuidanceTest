from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import time

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

while (True):
  mem = input('')
  starttime = time.time()
  with open('preprompt.txt', 'r') as file:
    prompt = file.read()
  sequences = pipeline(
    prompt,
    max_length=400,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
  )
  for seq in sequences:
    print(f"Result: {seq['generated_text']}")
  print(f'time: ${time.time() - starttime}')