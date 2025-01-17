# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# input = 'What is your name?'
# input = 'Summarize this story: In a small village surrounded by hills and forests, a curious young man named Taojie stood out. Unlike others content with routine, Taojie dreamed of exploring distant lands, inspired by travelers’ tales. One crisp autumn morning, drawn by rumors of a hidden treasure in the forbidden forest, he decided to venture into the unknown despite elders’ warnings of danger.'
intput = 'I have three numbers: 5, 8, and 12. The sum of the first two numbers is 13, and the sum of the second and third numbers is 20. What is the sum of all three numbers?'

input_ids = tokenizer.encode(input, return_tensors='pt')
output = model.generate(input_ids, max_length=1000, num_return_sequences=1, do_sample=True, temperature=0.85)

print(output)
print(tokenizer.decode(output[0], skip_special_tokens=True))
