import guidance
import time

# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.Transformers('mosaicml/mpt-1b-redpajama-200b-dolly', device=0)

while (True):
  mem = input('')
  starttime = time.time()
  with open('prompt.txt', 'r') as file:
    prompt = file.read()
  program = guidance(prompt)
  executed_program = program(memory=mem)
  print(executed_program)
  print(f'time: ${time.time() - starttime}')