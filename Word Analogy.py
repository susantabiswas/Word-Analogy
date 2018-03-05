
# coding: utf-8

# # <u>Word Analogy using Word Embeddings 
# In the word analogy task, we have $"a is to b as c is to __"$. For example is 'boy is to girl as king is to queen' .
# 
# We find a word $d$, such that the associated word vectors $e_a, e_b, e_c, e_d$ are related in the following manner: $e_b - e_a \approx e_d - e_c$. <br>
# For finding $d$ we measure the similarity between $e_b - e_a$ and $e_d - e_c$ using **cosine** similarity. 

import numpy as np
from utility import *


# For this task we will be using GLoVe Word Embeddings. Let us load that up.
# We will have
# - `words`: set of words in the vocabulary.
# - `word_to_vec`: dict mapping words to their GloVe vector representation.

words, word_to_vec = load_glove_vectors('data/glove.6B.50d.txt')


# ### Approach
# We will be using **cosine** similarity for finding the suitable word. We will use $e_b - e_a$ and $e_d - e_c$ as the two vectors to find their cosine, where $e_d$ is searched from all the other words in the vocabulary.
# 
# Given two vectors $u$ and $v$, cosine similarity is defined as follows: 
# 
# $$\text{Cosine Similarity(u, v)} = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta) $$
# 
# where $u.v$ is the dot product of two vectors, $||u||_2$ is the norm of the vector $u$, and $\theta$ is the angle between $u$ and $v$.
# <br>Norm of $u$ is defined as $ ||u||_2 = \sqrt{\sum_{i=1}^{n} u_i^2}$
# 
# This similarity depends on the angle between $u$ and $v$.
# <br>If $u$ and $v$ are very similar, their cosine similarity will be close to 1<br>
# If they are dissimilar, the cosine similarity will take a smaller value. 

# finds the cosine similarity between u and v
'''
    Arguments:
        u(n,) - vector of words            
        v(n,) - vector of words 
    Returns:
        cosine_sim - the cosine similarity between u and v
'''
def find_cosine_similarity(u, v):
    distance = 0.0
    
    # find the dot product between u and v 
    dot = np.dot(u,v)
    # find the L2 norm of u 
    norm_u = np.sqrt(np.sum(u**2))
    # Compute the L2 norm of v
    norm_v = np.sqrt(np.sum(v**2))
    # Compute the cosine similarity
    cosine_sim = dot/(norm_u)/norm_v
    
    return cosine_sim


# does the Word analogy task: a is to b as c is to ____
def find_analogy(word_a, word_b, word_c, word_to_vec):
    # convert words to lower case
    word_a = word_a.lower()
    word_b = word_b.lower()
    word_c = word_c.lower()
    
    
    # find the word embeddings for word_a, word_b, word_c
    e_a, e_b, e_c = word_to_vec[word_a], word_to_vec[word_b], word_to_vec[word_c]
    
    words = word_to_vec.keys()
    max_cosine_sim = -999              
    best_word = None                  

    # search for word_d in the whole word vector set
    for w in words:        
        # ignore input words
        if w in [word_a, word_b, word_c] :
            continue

        # Compute cosine similarity between the vectors u and v
        #u:(e_b - e_a) 
        #v:((w's vector representation) - e_c)
        cosine_sim = find_cosine_similarity(e_b - e_a, word_to_vec[w] - e_c)
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            # update word_d
            best_word = w
        
    return best_word



# for taking input from the user and doing word analogy task on that
def take_input():
    print('a --> b :: c --> d')
    print('Enter a, b, c words separated by space')
    words = input().split(' ')
    
    best_pick = find_analogy(*words, word_to_vec)
    print ('{} -> {} :: {} -> {}'.format( *words, best_pick))
    print('Best pick: ' + best_pick)


def main():
    take_input()

if __name__ == main():
    main()
