import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import pydot
import cPickle as pickle
import os as os
import utils
input_mask_mode = 'sentence'
vocab = {}
ivocab = {}
word_vector_size = 50
word2vec = utils.load_glove(word_vector_size)
"""
tasks = []
task = None
q_idx = []
for i, line in enumerate(open("s_data.txt")):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "SF": ""} 
	    q_idx = []      
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        if line.find('?') == -1:
            task["C"] += line
        else:
	    q_idx.append(id-1)
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
	    SF = tmp[2].strip()
	    SF = SF.split(" ")
	    SF = np.asarray(list((int(a)-1) for a in SF))
	    for i in reversed(q_idx):
	    	SF = [(a-1 if i<a else a) for a in SF]
	    s = []
	    for i in range(0,7):
		if i<len(SF):
	    		s.append(SF[i])
		else:
			s.append(SF[len(SF)-1])
	    s = np.asarray(s)
	    task["SF"] = s
            tasks.append(task.copy())
"""
def _process_input(data_raw):
        questions = []
        inputs = []
        answers = []
        input_masks = []
	supp_fact = []

        for x in data_raw:
            inp = x["C"].lower().split(' ') 
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            sf = x["SF"]
	    SF = []
	    for i in range(0,7):
		if i<len(sf):
	    		SF.append(sf[i])
		else:
			SF.append(sf[len(sf)-1])
            inp_vector = [utils.process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = word_vector_size, 
                                        to_return = "word2vec") for w in inp]
                                        
            q_vector = [utils.process_word(word = w, 
                                        word2vec = word2vec, 
                                        vocab = vocab, 
                                        ivocab = ivocab, 
                                        word_vector_size = word_vector_size, 
                                        to_return = "word2vec") for w in q]
            
            inputs.append(np.vstack(inp_vector))
            questions.append(np.vstack(q_vector))
	    supp_fact.append(SF)
            answers.append(utils.process_word(word = x["A"], 
                                            word2vec = word2vec, 
                                            vocab = vocab, 
                                            ivocab = ivocab, 
                                            word_vector_size = word_vector_size, 
                                            to_return = "index"))
            # NOTE: here we assume the answer is one word! 
            if input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) 
            elif input_mask_mode == 'sentence': 
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32)) 
            else:
                raise Exception("invalid input_mask_mode")
        inputs = np.asarray(inputs)
	questions = np.asarray(questions)
	answers = np.asarray(answers)
	input_masks = np.asarray(input_masks)
	supp_fact = np.asarray(supp_fact)
        return inputs, questions, answers, input_masks, supp_fact

babi_train_raw, babi_test_raw = utils.get_babi_raw("1","1")
train_input, train_q, train_answer, train_input_mask, train_sf = _process_input(babi_train_raw)
a = train_sf[0]
print a
b = np.array([1,2,3])
print b[2]
