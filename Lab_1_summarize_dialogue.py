
# Load the datasets, Large Language Model (LLM), and configurator.

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig


"""
2 - Summarize Dialogue without Prompt Engineering

In this use case, you will be generating a summary of a dialogue 
with the pre-trained Large Language Model (LLM) FLAN-T5 from Hugging Face. 
"""
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)


print("="*100)
print("LAB 1 - **Summarize Dialogue without Prompt Engineering**")
print("="*100)

print("\n")
print("-"*75)
print("\t\tPrint a couple of dialogues with their baseline summaries")
example_indices = [40, 200]
dash_line = '_'.join('' for x in range(100))


for i, index in enumerate(example_indices):
    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print('INPUT DIALOGUE:')
    print(dataset['test'][index]['dialogue'])
    print(dash_line)
    print('BASELINE HUMAN SUMMARY:')
    print(dataset['test'][index]['summary'])
    print(dash_line)
    print()


# Load the FLAN-T5 model, creating an instance of the AutoModelForSeq2SeqLM class with the .from_pretrained() method.
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# To perform encoding and decoding, you need to work with text in a tokenized form.
# Tokenization is the process of splitting texts into smaller units that can be processed by the LLM models.
# Download the tokenizer for the FLAN-T5 model using AutoTokenizer.from_pretrained() method.
# Parameter use_fast switches on fast tokenizer.

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Test the tokenizer encoding and decoding a simple sentence
print("-"*75)
print("\t\tTest the tokenizer encoding and decoding a simple sentence")
sentence = "What time is it, Tom?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')

sentence_decoded = tokenizer.decode(
    sentence_encoded["input_ids"][0],
    skip_special_tokens=True
)

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print("\nDECODED SENTENCE:")
print(sentence_decoded)


# Now it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering.
# Prompt engineering is an act of a human changing the prompt (input) to improve the response for a given task.
print("-"*75)
print("\t\tNow it's time to explore how well the base LLM summarizes a dialogue without any prompt engineering.")

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')


"""
3 - Summarize Dialogue with an Instruction Prompt

Prompt engineering is an important concept in using foundation models for text generation. 
"""

# 3.1 – Zero Shot Inference with an Instruction Prompt
# In order to instruct the model to perform a task - summarize a dialogue - you can take the dialogue
# and convert it into an instruction prompt. This is often called zero shot inference.

print("-"*75)
print("\t\t3.1 - Zero Shot Inference with an Instruction Prompt")

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
        Summarize the following conversation.
        
        {dialogue}
        
        Summary:
        """

    # Input constructed prompt instead of the dialogue.
    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')


# 3.2 – Zero Shot Inference with Prompt Template from FLAN-T5
# Let's use a slightly different prompt. FLAN-T5 has many prompt templates that are published for certain tasks

print("-"*75)
print("\t\t3.2 - Zero Shot Inference with Prompt Template from FLAN-T5")
for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    prompt = f"""
        Dialogue:
        
        {dialogue}
        
        What was going on?
        """

    inputs = tokenizer(prompt, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(dash_line)
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{prompt}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}\n')

# Notice that this prompt from FLAN-T5 did help a bit, but still struggles to
# pick up on the nuance of the conversation. This is what you will try to solve with the few shot inferencing.

"""
4 - Summarize Dialogue with One Shot and Few Shot Inference

One shot and few shot inference are the practices of providing an LLM with either one 
or more full examples of prompt-response pairs that match your task - before your actual prompt that 
you want completed. This is called "in-context learning" and 
puts your model into a state that understands your specific task.
"""

# 4.1 - One Shot Inference
# Let's build a function that takes a list of example_indices_full,
# generates a prompt with full examples, then at the end appends the prompt which
# you want the model to complete (example_index_to_summarize).

print("-"*75)
print("\t\t4.1 - One Shot Inference")


def make_prompt(example_indices_full, example_index_to_summarize):
    prompt = ''
    for index in example_indices_full:
        dialogue = dataset['test'][index]['dialogue']
        summary = dataset['test'][index]['summary']

        # The stop sequence '{summary}\n\n\n' is important for FLAN-T5. Other models may have their own preferred stop sequence.
        prompt += f"""
            Dialogue:

            {dialogue}

            What was going on?
            {summary}


            """

    dialogue = dataset['test'][example_index_to_summarize]['dialogue']

    prompt += f"""
        Dialogue:

        {dialogue}

        What was going on?
        """

    return prompt


# Construct the prompt to perform one shot inference
example_indices_full = [40]
example_index_to_summarize = 200

one_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(one_shot_prompt)


# Now pass this prompt to perform the one shot inference:
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(one_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ONE SHOT:\n{output}')

# 4.2 - Few Shot Inference
# Let's explore few shot inference by adding two more full dialogue-summary pairs to your prompt.

print("-"*75)
example_indices_full = [40, 80, 120]
example_index_to_summarize = 200

few_shot_prompt = make_prompt(example_indices_full, example_index_to_summarize)

print(few_shot_prompt)

# Now pass this prompt to perform a few shot inference:
summary = dataset['test'][example_index_to_summarize]['summary']

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')


"""
5 - Generative Configuration Parameters for Inference

You can change the configuration parameters of the generate() method to see a different output from the LLM. 
So far the only parameter that you have been setting was max_new_tokens=50, which defines the maximum number of tokens to generate.

A convenient way of organizing the configuration parameters is to use  `GenerationConfig` class.


Putting the parameter do_sample = True, you activate various decoding strategies which influence 
the next token from the probability distribution over the entire vocabulary. You can then adjust 
the outputs changing temperature and other parameters (such as top_k and top_p).
"""

generation_config = GenerationConfig(max_new_tokens=50)
# generation_config = GenerationConfig(max_new_tokens=10)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.1)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.5)
# generation_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=1.0)

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
output = tokenizer.decode(
    model.generate(
        inputs["input_ids"],
        generation_config=generation_config,
    )[0],
    skip_special_tokens=True
)

print(dash_line)
print(f'MODEL GENERATION - FEW SHOT:\n{output}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')


# Comments related to the choice of the parameters in the code cell above:
# Choosing max_new_tokens=10 will make the output text too short, so the dialogue summary will be cut.
# Putting do_sample = True and changing the temperature value you get more flexibility in the output.
