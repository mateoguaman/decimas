import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, pipeline
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM
from random import randint
from tqdm import tqdm

def test_gemma():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

    input_text = "Creating a décima requires careful attention to both its structure and rhyme. A décima is a ten-line stanza of poetry that is octosyllabic (each line has eight syllables) and typically follows a specific rhyme scheme, which is often ABBAACCDDC. Let's create a décima that adheres to these guidelines."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids, max_new_tokens=200)
    print(tokenizer.decode(outputs[0]))

def main():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2" #"mistralai/Mixtral-8x7B-Instruct-v0.1"  #"mistralai/Mistral-7B-v0.1"
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     device_map="auto",
    #     torch_dtype=torch.float16
    # )
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load our test dataset
    eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
    rand_idx = randint(0, len(eval_dataset)-1)

    # Test on sample
    prompt = eval_dataset[rand_idx]["messages"][0]
    outputs = pipe(prompt['content'], max_new_tokens=512, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    # prompt = pipe.tokenizer.apply_chat_template(eval_dataset[rand_idx]["messages"][:2], tokenize=False, add_generation_prompt=True)
    # outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)

    print(outputs[0]['generated_text'])

    print(f"Query:\n{eval_dataset[rand_idx]['messages'][0]['content']}")
    print(f"Original Answer:\n{eval_dataset[rand_idx]['messages'][1]['content']}")
    print("----------")
    print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt['content']):].strip()}")
    print("----------")

if __name__ == "__main__":
    test_gemma()