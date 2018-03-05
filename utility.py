import numpy as np 

# for loading the glove word embedding matrix values
def load_glove_vectors(glove_file):
    with open(glove_file, 'r', encoding="utf-8") as file:
        # unique words
        words = set()
        word_to_vec = {}
        # each line starts with a word then the values for the different features
        for line in file:
            line = line.strip().split()
            # take the word 
            curr_word = line[0]
            words.add(curr_word)
            # rest of the features for the word
            word_to_vec[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec

