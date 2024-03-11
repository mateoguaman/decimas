import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, pipeline
from trl import SFTTrainer, setup_chat_format
from peft import LoraConfig, AutoPeftModelForCausalLM
from random import randint
from tqdm import tqdm

def dataset_loader(data_dir):
    texts = []
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
                # texts.append(f.read().replace('\n', ' '))
                texts.append(f.read())
    
    # Create a dataset from the list of texts
    dataset = Dataset.from_dict({'text': texts}, split="train")
    return dataset

# def create_instruction_dataset(sample):
#   prompt = "Creating a décima requires careful attention to both its structure and rhyme. A décima is a ten-line stanza of poetry that is octosyllabic (each line has eight syllables) and typically follows a specific rhyme scheme, which is often ABBAACCDDC. Let's create a décima that adheres to these guidelines."
#   return {
#     "prompt": prompt,
#     "completion": sample["text"]
#   }


def create_conversation(sample):
  system_message = "" #"""You are a generator of Decimas. Users will ask you to generate decimas, and you will respond with only the poem in Spanish. Be succint, and do not repeat yourself."""
  prompt = """Creating a décima requires careful attention to both its structure and rhyme. A décima is a ten-line stanza of poetry that is octosyllabic (each line has eight syllables) and typically follows a specific rhyme scheme, which is often ABBAACCDDC. Let's create a décima that adheres to these guidelines."""
  return {
    "messages": [
    #   {"role": "system", "content": system_message},
      {"role": "user", "content": prompt},
      {"role": "assistant", "content": sample["text"]}
    ]
  }

def evaluate(pipe, sample):
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    predicted_answer = outputs[0]['generated_text'][len(prompt):].strip()
    if predicted_answer == sample["messages"][2]["content"]:
        return 1
    else:
        return 0

def main():
    ## ---------- Create dataset if it doesn't exist yet ---------- 
    seed = 42
    if not(os.path.exists("train_dataset.json") and os.path.exists("test_dataset.json")):
        dataset = dataset_loader('./data')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
        dataset = dataset.train_test_split()
        
        # save datasets to disk
        dataset["train"].to_json("train_dataset.json", orient="records")
        dataset["test"].to_json("test_dataset.json", orient="records")
    
    ## Load dataset
    dataset = load_dataset("json", data_files="train_dataset.json", split="train")

    ## ---------- Setup model and tokenizer ---------- 

    # Hugging Face model id
    model_id = "mistralai/Mistral-7B-Instruct-v0.2" # "mistralai/Mistral-7B-v0.1" # or "codellama/CodeLlama-7b-hf" # or `mistralai/Mistral-7B-v0.1`

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
    tokenizer.padding_side = 'right' # to prevent warnings

    # # set chat template to OAI chatML, remove if you start from a fine-tuned model
    model, tokenizer = setup_chat_format(model, tokenizer)

    ## ---------- Setup LoRA and training arguments ----------

    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=f"{model_id}-decimas", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=3,          # batch size per device during training
        gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=1,                        # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        report_to="tensorboard",                # report metrics to tensorboard
    )

    ## ---------- Setup Supervised Fine-Tuning Trainer ----------

    max_seq_length = 3072 # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False, # No need to add additional separator token
        }
    )

    ## ---------- Train ----------

    trainer.train()
    # save model
    trainer.save_model()

    #### COMMENT IN TO MERGE PEFT AND BASE MODEL ####

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")

    ## ---------- Test ----------

    peft_model_id = args.output_dir

    # Load Model with PEFT adapter
    model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    # model = AutoModelForCausalLM.from_pretrained(
    #     peft_model_id,
    #     device_map="auto",
    #     torch_dtype=torch.float16
    # )
    # tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

    # load into pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load our test dataset
    eval_dataset = load_dataset("json", data_files="test_dataset.json", split="train")
    rand_idx = randint(0, len(eval_dataset))

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

    
    ## Evaluate

    # success_rate = []
    # number_of_eval_samples = 21
    # # iterate over eval dataset and predict
    # for s in tqdm(eval_dataset.shuffle().select(range(number_of_eval_samples))):
    #     success_rate.append(evaluate(pipe,s))

    # # compute accuracy
    # accuracy = sum(success_rate)/len(success_rate)

    # print(f"Accuracy: {accuracy*100:.2f}%")


    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()