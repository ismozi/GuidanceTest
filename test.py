import guidance
import time

# set the default language model used to execute guidance programs
# mpt = guidance.llms.transformers.MPT('mosaicml/mpt-7b-instruct', device=0)
openllama = guidance.llms.transformers.LLaMA('openlm-research/open_llama_3b_600bt_preview', device=0)
# incite = openllama = guidance.llms.Transformers('togethercomputer/RedPajama-INCITE-Instruct-3B-v1', device=0)

while (True):
  mem = input('')
  starttime = time.time()
  with open('prompt.txt', 'r') as file:
    prompt = file.read()
  program = guidance(prompt)
  executed_program = program(memory=mem, llm=openllama)
  print(executed_program)
  print(f'time: ${time.time() - starttime}')