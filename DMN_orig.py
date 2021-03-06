import numpy as np
import theano
import theano.tensor as T
import os as os
import utils
import sys
import updates as upd
import init
import random
import cPickle as pickle

vocab = {}
ivocab = {}
word_vector_size = 50
word2vec = utils.load_glove(word_vector_size)
input_mask_mode = 'sentence'
answer_tm = 3
task = "7"
hops = 3
m = 0.02


def softmax(x):
    e_x = T.exp(x - x.max(axis=0, keepdims=True))
    out = e_x / e_x.sum(axis=0, keepdims=True)
    return out
def constant_param(value=0.0, shape=(0,)):
    return theano.shared(init.Constant(value).sample(shape), borrow=True)
   
def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(init.Normal(std, mean).sample(shape), borrow=True)

def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])

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
	    for i in range(0,hops):
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

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tm1
	return s_t

class DMN(object):
	def __init__(self,hidden_dim,vocab_size,batch_size = 20,bptt_truncate=-1):
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.bptt_truncate = bptt_truncate
		self.tm = hops
		self.batch_size = batch_size
		#self.gru_p = GRUblock(hidden_dim,1)
		self.U0_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.U1_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.U2_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W0_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_i = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_i = constant_param(value=m, shape=(self.hidden_dim,))
		self.b1_i = constant_param(value=m, shape=(self.hidden_dim,))
		self.b2_i = constant_param(value=m, shape=(self.hidden_dim,))
		

		#self.gru_em = GRUblock(hidden_dim,1)	
		self.U0_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.U1_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.U2_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W0_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_em = normal_param(std=m, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_em = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_em = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_em = constant_param(value=0.0, shape=(self.hidden_dim,))
			
		self.W1 = normal_param(std=0.0028, shape=(5*hidden_dim, (7*hidden_dim)+2))
		self.W2 = normal_param(std=0.04, shape=(1,5*hidden_dim))
		self.b1 = constant_param(value=0.0, shape=(5*hidden_dim,))
		self.b2 = constant_param(value=0.0, shape=(1,))
		self.Wa = normal_param(std=0.02, shape=(vocab_size,hidden_dim))
		self.Wb = normal_param(std=0.02, shape=(hidden_dim,hidden_dim))
		self.U_ans0 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.U_ans1 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.U_ans2 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.W_ans0 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim))
		self.W_ans1 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim))
		self.W_ans2 = normal_param(std=m, shape=(self.hidden_dim,self.hidden_dim))
		self.b_ans0 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b_ans1 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b_ans2 = constant_param(value=0.0, shape=(self.hidden_dim,))

		p = T.matrix()
		q = T.matrix()
		marked = T.ivector()
		y = T.iscalar()
		sf = T.ivector()
		s_p,updates = theano.scan(self.input_next_state,sequences=p,		
						    outputs_info=T.zeros_like(self.b2_i))
		s_q,updates = theano.scan(self.input_next_state,sequences=q,
							outputs_info=T.zeros_like(self.b2_i))
		c = s_p.take(marked,axis=0)
		q_q = s_q[-1]
		
		epm,epm_updates = theano.scan(fn = self.step_em,non_sequences=[c,q_q], outputs_info=[None,q_q.copy()],n_steps=self.tm) 
		last_mem = epm[1][-1]
		self.gate = epm[0] 
		self.shit = theano.function([p,q,marked],[self.gate])
		self.loss_gates = T.mean(T.nnet.categorical_crossentropy(self.gate, sf))
		pred = T.nnet.softmax(T.dot(self.Wa,last_mem))[0]
		results,r_updates = theano.scan(self.answer_step,non_sequences=[q_q],outputs_info=[last_mem,pred],n_steps=answer_tm)		
		self.prediction = results[1][-1]	
		self.loss_ans = T.nnet.categorical_crossentropy(self.prediction.dimshuffle('x', 0), T.stack([y]))[0] 
		self.loss_ce = (self.loss_ans) + self.loss_gates
		#another scan for batch training : sequences = data , loss gets added each step
		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i,
				self.U0_em,self.W0_em,self.b0_em,
				self.U1_em,self.W1_em,self.b1_em,
				self.U2_em,self.W2_em,self.b2_em,
				self.W1,self.W2,self.b1,self.b2,self.Wb]
		self.params = self.params + [self.U_ans0,self.W_ans0,self.b_ans0,self.U_ans1,self.W_ans1,self.b_ans1,
							self.U_ans2,self.W_ans2,self.b_ans2,self.Wa]
		#loss_ce = loss_ce + l2_reg(self.params)
		updts = upd.adadelta(self.loss_ce,self.params)
		self.train_fn = theano.function(inputs=[p,q,marked,y,sf],outputs=[self.prediction,self.loss_ce],updates = updts,allow_input_downcast = True)
		self.f = theano.function([p,q,marked,y],[self.loss_ans,self.prediction])

	def input_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_i,self.W0_i,self.b0_i,self.U1_i,self.W1_i,self.b1_i,self.U2_i,self.W2_i,self.b2_i)
		return s_t

	def em_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_em,self.W0_em,self.b0_em,self.U1_em,self.W1_em,self.b1_em,self.U2_em,self.W2_em,
					self.b2_em)
		return s_t

	def new_attn_step(self,c_t,g_tm1,m_im1,q):
		cWq = T.stack([T.dot(T.dot(c_t, self.Wb), q)])
        	cWm = T.stack([T.dot(T.dot(c_t, self.Wb), m_im1)])
		z = T.concatenate([c_t,m_im1,q,c_t*q,c_t*m_im1,T.abs_(c_t-q),T.abs_(c_t-m_im1),cWq,cWm],axis=0)
		l_1 = T.dot(self.W1, z) + self.b1
		l_1 = T.tanh(l_1)
		l_2 = T.dot(self.W2,l_1) + self.b2
		return l_2[0]

	def step_ce(self,a,b):
		return T.mul(b,a)

	def new_episode(self,c,mem,q):
		g, g_updates = theano.scan(fn=self.new_attn_step,
		    sequences=c,
		    non_sequences=[mem,q],
		    outputs_info=T.zeros_like(c[0][0])) 

		gs = T.nnet.softmax(g)[0]
	
		w,w_updates = theano.scan(fn=self.step_ce,sequences = [gs,c],outputs_info = None)
		e = T.sum(w,axis=0)

		return gs,e

	def step_em(self,m_tm1,c,q):
		G,current_episode = self.new_episode(c,m_tm1,q)
		m_t = self.em_next_state(current_episode,m_tm1)
		return G,m_t

	def answer_step(self,a_tm1,pred_tm1,q):
		conc = T.concatenate([pred_tm1,q])
		a_t = gru_next_state(conc,a_tm1,self.U_ans0,self.W_ans0,self.b_ans0,self.U_ans1,self.W_ans1,self.b_ans1,self.U_ans2, self.W_ans2,self.b_ans2)
		pred = T.nnet.softmax(T.dot(self.Wa,a_t))[0]
		return a_t,pred

	def save_params(self, file_name, epoch):
		with open(file_name, 'w') as save_file:
		    pickle.dump(
		        obj = {
		            'params' : [x.get_value() for x in self.params],
		            'epoch' : epoch, 
		        },
		        file = save_file,
		        protocol = -1
		    )
    
	def load_state(self, file_name):
		print "==> loading state %s" % file_name
		with open(file_name, 'r') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)

	def train(self,tr_input,tr_q,tr_ans,tr_mask,tr_sf):
		l = len(tr_input)
		print "starting..."
		for j in range(0,100):
			a_loss = 0.0
			tr_input,tr_q,tr_ans,tr_mask,tr_sf = shuffle(tr_input,tr_q,tr_ans,tr_mask,tr_sf)
			for i in range(0,l):
				pred,loss = self.train_fn(tr_input[i],tr_q[i],tr_mask[i],tr_ans[i],tr_sf[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f"%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states_orig/states'+task+'/DMN_orig.epoch%d' %(j)
					self.save_params(fname,j)
		
	def test(self,test_inp,test_q,test_ans,test_mask):
		l = len(test_inp)
		print "testing..."
		y_true = []
		y_pred = []
		a_loss = 0
		for i in range(0,l):
			loss,pred = self.f(test_inp[i],test_q[i],test_mask[i],test_ans[i]) 
			a_loss = a_loss + loss		
			y_true.append(test_ans[i])
			y_pred.append(pred.argmax(axis=0))
		accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
		print "accuracy: %.2f percent" % (accuracy * 100.0 / l)

def shuffle(train_input,train_q,train_answer,train_input_mask,train_sf):
        print "==> Shuffling the train set"
        combined = zip(train_input,train_q,train_answer,train_input_mask,train_sf)
        random.shuffle(combined)
        train_input, train_q, train_answer, train_input_mask, train_sf = zip(*combined)
	return train_input,train_q,train_answer,train_input_mask,train_sf
	
babi_train_raw, babi_test_raw = utils.get_babi_raw(task,task)
train_input, train_q, train_answer, train_input_mask, train_sf = _process_input(babi_train_raw)
test_input, test_q, test_answer, test_input_mask, test_sf = _process_input(babi_test_raw)
vocab_size = len(vocab)

a1 = train_input
a2 = train_q
a3 = train_input_mask
a4 = train_answer	
dmn = DMN(word_vector_size,vocab_size)
dmn.load_state('states_orig/states7/DMN_orig.epoch20')
#dmn.train(a1,a2,a4,a3,train_sf)
dmn.test(a1,a2,a4,a3)
dmn.test(test_input,test_q,test_answer,test_input_mask)




		
