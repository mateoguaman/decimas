# Decimas

## Installation Instructions

**Note:** This package was tested and run on Ubuntu 22.04 using two Nvidia 4090s.

```bash
conda create -n decimas python=3.10
conda activate decimas
pip install "torch==2.1.2" tensorboard
pip install  --upgrade   "transformers==4.36.2"   "datasets==2.16.1"   "accelerate==0.26.1"   "evaluate==0.4.1"   "bitsandbytes==0.42.0"
pip install git+https://github.com/huggingface/trl@a3c5b7178ac4f65569975efadc97db2f3749c65e --upgrade
pip install git+https://github.com/huggingface/peft@4a1559582281fc3c9283892caea8ccef1d6f5a4f --upgrade
pip install ninja packaging
conda install nvidia::cuda-nvcc
MAX_JOBS=4 pip install flash-attn --no-build-isolation

```

To train and generate a Decima, simply run:

```bash
python finetune.py
```

For this project, I mostly followed instructions from this [tutorial](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#5-test-and-evaluate-the-llm).

Here are examples of the generated Decimas before and after fine-tuning ```mistralai/Mistral-7B-Instruct-v0.2```:

**Prompt:** 

```Creating a décima requires careful attention to both its structure and rhyme. A décima is a ten-line stanza of poetry that is octosyllabic (each line has eight syllables) and typically follows a specific rhyme scheme, which is often ABBAACCDDC. Let's create a décima that adheres to these guidelines.```

**Pre-trained:**

```
First, let's choose a topic or theme for our poem. For this example, let's write about the beauty of nature.

Now, let's begin writing our décima:

1. In fields of green, where flowers bloom,
2. The gentle breeze, a soothing balm,
3. A symphony of life, a sweet perfume,
4. The sun sets, a day's end, calm and calm,
5. A canvas painted with the sky's hue,
6. The stars emerge, a night's gentle view,
7. A tranquil lake, a mirror true,
8. Reflecting nature's beauty, pure and new,
9. The moon's soft glow, a peaceful cue,
10. In nature's arms, our worries subdue.

In this example, we've followed the octosyllabic structure and the ABBAACCDDC rhyme scheme. The décima we've created is about the beauty of nature and the peace it brings. Remember, writing a décima requires careful attention to both its structure and rhyme, but the end result can be a beautiful and meaningful poem.
```

**Fine-tuned:**

```
de mi vida

Ayer, cuando era niño,
me llamaron a la escuela
y me dijeron:
"El que no sabe leer
no puede ser hombre."

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.
Yo me sentí herida
porque mi madre no sabía
ni leer ni escribir.

Yo me sentí

```

The fine-tuning does not get it to produce decimas yet, but at least it gets the model to produce something that looks like a poem in Spanish. Next steps involve doing an additional fine-tuning stage with a larger dataset of poems in Spanish and then fine-tuning on the smaller decimas dataset included here.