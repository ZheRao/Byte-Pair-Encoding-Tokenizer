import os
import shutil
from typing import Optional
import pickle
import numpy as np
from tqdm.auto import tqdm

class ConfigError(Exception):
    """for invalid user configuration of tokenizer class"""
    pass

class base_tokenizer:
    """
        This is the base class for tokenizer, it provde basic functionalities such as produce pair counts and merge new tokens
        Args:

    """
    def __init__(self):
        # initialize base vocabulary dictionary which is the character encoding based on UTF-8
        self._base_vocab = {i: bytes([i]) for i in range(256)}
        self._base_vocab_size = 256
        
    
    def _get_file_paths(self,title,vocab_size,tokenizer_folder_path=os.getcwd()):
        self.folder = title + "_base_tok_folder"
        self.vocab_path = os.path.join(tokenizer_folder_path,self.folder, title + "_vocab_dict_size"+str(vocab_size)+".pkl")
        self.merges_path = os.path.join(tokenizer_folder_path,self.folder, title + "_merge_history_size"+str(vocab_size)+".pkl")
        self.tokens_path = os.path.join(tokenizer_folder_path,self.folder, title + "_tokens_size"+str(vocab_size)+".npy")

    def _get_pair_counts(self,tokens):
        """
            treverse through the entire encoded text, produce a dictionary with paired occurrences of adjacent tokens
                key: token pairs, e.g., (106, 32)
                value: counts of occurrence of key, e.g., 300
                meaning: token pair (106, 32) occurred 300 times in the text
        """

        count_dict = {}
        for (c1, c2) in zip(tokens[:-1],tokens[1:]):
            count_dict[(c1,c2)] = count_dict.get((c1,c2),0) + 1
        return count_dict
    
    def _merge_pair(self,tokens,pair,new_token):
        """
            Replace all occurrences of pair in tokens by new_token
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i],tokens[i+1]) == pair:
                new_tokens.append(new_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def _retrieve_training_history(self,title,vocab_size):
        """
            retrieve the dictionaries
        """
        self._get_file_paths(title,vocab_size)
        try:
            with open(self.vocab_path,"rb") as f:
                vocab = pickle.load(f)
            with open(self.merges_path,"rb") as f:
                merges = pickle.load(f)
            with open(self.tokens_path,"rb") as f:
                past_tokens = np.load(f)
        except:
            m = f"Dictionary files do not exit, tokenizer requires training with {title} dataset. \nOr provided with inconsistent vocab_size, use os.listdir to inspect dictionary files."
            m_more = " Or past_tokens cannot be retreived, if this is the case, encode text first"
            raise FileNotFoundError(m+m_more)
        else:
            return vocab, merges, past_tokens

class TrainTokenizer(base_tokenizer):
    """
        This class implement compression algorithm described in 
            https://en.wikipedia.org/wiki/Byte_pair_encoding#:~:text=Byte%20pair%20encoding%20(also%20known,for%20use%20in%20downstream%20modeling
        It takes text and title, train a tokenizer and store the files in a directory
        Args:
            text: str, actual text
            title: str, name of the mateirals that the tokenizer is training on
            fresh_start: bool, whether to train from scratch or continue training/compressing, default=True
            final_vocab_size: int, final vocabulary size after compression - determines how many merges to perform, in thousands
            last_vocab_size: int, if continue training, what is the last final_vocab_size in thousands: 10 -> 10,000
        
        Folder/Title Naming Convention:
            "book_title_base_tok_folder" - all lower case, connected with underscore
        
        Sub-files (in the _tok_folder) Naming Convetion:
            "book_title_vocab_dict_size10.pkl" stores encoding dictionary, where 10 means 10,000 vocabulary size
            "book_title_merge_history_size10.pkl" stores merging history, where 10 means 10,000 vocabulary size
            "book_title_tokens_size10.npy" stores tokens from last compression, with tokenization with size 10,000

        
    """

    def __init__(self, text: str, title: str, final_vocab_size: int =6000, fresh_start: bool =True, last_vocab_size: Optional[int] =None):
        super(TrainTokenizer,self).__init__()
        self.title = title
        self.final_vocab_size = final_vocab_size
        # initialize training vocabulary and merge history dictionaries
        if fresh_start:
            self.vocab = self._base_vocab
            self.merge_history = {}
            self.tokens = list(text.encode("utf-8"))
        else:
            if last_vocab_size is None: raise ConfigError("for continue training (fresh_start == False), last final_vocab_size must be provided")
            if final_vocab_size <= last_vocab_size: raise ConfigError("unable to perform tokenizer training, because new vocabulary size must be larger than previous vocabulary size")
            self.vocab, self.merge_history, self.tokens = self._retrieve_training_history(self.title,last_vocab_size)
        assert len(self.vocab) == (len(self.merge_history) + self._base_vocab_size), "dictionary lengths not matching - the following should be true: voca = merge_hist + 256"
        assert final_vocab_size > len(self.vocab), f"final vocabulary size specified is too small, must be larger than {len(self.vocab)}"
    
    def _perform_merge(self):
        """
            Training loop compression process:
                1. identify top pair
                2. swap the occurrences of top pair in the original tokens by new token
                3. update merge_history and vocab
            after training, vocab, merge_history are saved as pickle files and final tokens are saved as npy file
        """
        vocab_size = len(self.merge_history) + self._base_vocab_size
        num_merges = self.final_vocab_size - vocab_size
        progress_bar = tqdm(range(num_merges))

        for i in range(num_merges):
            progress_bar.update(1)
            pair_counts = self._get_pair_counts(tokens=self.tokens)
            top_pair = max(pair_counts,key=pair_counts.get)
            self.tokens = self._merge_pair(tokens=self.tokens,pair=top_pair,new_token=vocab_size+i)
            self.merge_history[top_pair] = vocab_size+i
            self.vocab[vocab_size+i] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            #print(f"merged {top_pair} as {vocab_size+i}")
        
        self._get_file_paths(self.title,self.final_vocab_size)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        # save files to folder
        with open(self.vocab_path,"wb") as f:
            pickle.dump(self.vocab,f)
        with open(self.merges_path,"wb") as f:
            pickle.dump(self.merge_history, f)
        with open(self.tokens_path,"wb") as f:
            np.save(f,self.tokens)
    
    def run(self):
        self._perform_merge()

class ApplyTokenizer(base_tokenizer):
    """
        This subclass can encode and decode text based on trained tokenizer
        Arg:
            mode: str, "encode" or "decode"
            title: str, tokenizer trained on which texts user wish to apply
            vocab_size: int, which version of the tokenizer user wish to apply, usually in thousands
            tokenizer_folder_path: file path to the tokenizer folder, to access the dictionaries

    """
    def __init__(self, title, vocab_size, tokenizer_folder_path):
        super(ApplyTokenizer,self).__init__()
        self.vocab, self.merge_history, _ = self._retrieve_training_history(title=title,vocab_size=vocab_size,tokenizer_folder_path=tokenizer_folder_path)
    
    def encode(self,text):
        assert type(text) == str, "input for encoding is not string"
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            pairs = self._get_pair_counts(tokens=tokens) # dictionary of (101, 102) 30000
            pair_replace = min(pairs,key=lambda x: self.merge_history.get(x,float('inf'))) # if no match, returns itself
            if pair_replace not in self.merge_history:
                break
            new_token = self.merge_history[pair_replace]
            tokens = self._merge_pair(tokens=tokens,pair=pair_replace,new_token=new_token)
        return tokens

    def decode(self,tokens):
        text = b"".join([self.vocab[i] for i in tokens])
        text_decoded = text.decode("utf-8",errors="replace")
        return text_decoded

        


    

