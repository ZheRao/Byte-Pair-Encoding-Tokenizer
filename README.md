# Tokenizer
Implementing the Byte-Pair Encoding Tokenizer, used by GPT series, unfortunately, OpenAI doesn't allow tokenizer training, with this module, we can train our own tokenizer with our own text files

Improvement: adding regex to separate text into chunks, so merges will never happen across different elements (e.g., merge between "dog" and "!")
paper: 
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

In addition, invented a way to store list of list of integers (encoded token chuncks) as text into a text file (alternatively, saving it as numpy file .npy requires 10 times more space and 15 times more time to write and read file)
This is necessary for training tokenizer on a large corpus (e.g., 10 Million words), compression algorithm needs to run in succession instead of one run. Saving encoded token chunks avoid the necessity of encoding and decoding everytime when tokenizer is updated (encoding can be very expensive because many merges will take place)
for LOR, there were more than 800,000 chunks, saving and loading with npy format takes around 30 seconds in total, and 114 MB storage space
processing, saving, loading, and recovering with my method only took 3 seconds, and 14 MB storage space
Details are in improved_tokenizer_experiment.ipynb
