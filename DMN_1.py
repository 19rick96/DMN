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
answer_mode = 'feedforward'
answer_tm = 1
task = "3"


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
        return inputs, questions, answers, input_masks

class Layer(object):
	def __init__(self,U,b=None,activation=None):
		n_output,n_input = U.shape
		self.b = np.zeros(n_output,)
		if b!= None:
			assert b.shape == (n_output,)
		self.U = theano.shared(value=U.astype(theano.config.floatX),
					name='U',borrow=True)
		if b != None:
			self.b = theano.shared(value=b.reshape(1,n_output).astype(theano.config.floatX),
						name='b',borrow=True)
		self.activation = activation
		self.params = [self.U,self.b]
	def output(self,x):
		out = T.dot(self.U,x) + self.b
		out = out if self.activation is None else self.activation(out)
		return out[0]

class GRUblock(object):                           
	def __init__(self,dim,no_embed=0):
		self.dim = dim
		self.no_embed = no_embed

		E = np.random.uniform(-np.sqrt(1./dim), np.sqrt(1./dim), (dim,dim))
		self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
		"""
		U = np.random.uniform(-np.sqrt(1./dim), np.sqrt(1./dim), (3, dim, dim))
		W = np.random.uniform(-np.sqrt(1./dim), np.sqrt(1./dim), (3, dim, dim))
		b = np.zeros((3, dim))
		self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
		self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
		self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
		"""
		self.U0 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.U1 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.U2 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W0 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W1 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.W2 = normal_param(std=0.1, shape=(self.dim, self.dim))
		self.b0 = constant_param(value=0.0, shape=(self.dim))
		self.b1 = constant_param(value=0.0, shape=(self.dim))
		self.b2 = constant_param(value=0.0, shape=(self.dim))
	def calc_next_state(self,x_e,s_tm1):
		if self.no_embed == 0:
			x_t = self.E[:,x_e]
		else:
			x_t = x_e
		z_t = T.nnet.hard_sigmoid(self.U0.dot(x_t) + self.W0.dot(s_tm1) + self.b0)
		r_t = T.nnet.hard_sigmoid(self.U1.dot(x_t) + self.W1.dot(s_tm1) + self.b1)
	    	h_t = T.tanh(self.U2.dot(x_t) + self.W2.dot(s_tm1 * r_t) + self.b2)
	    	s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tm1
		o_t = T.zeros_like(s_t)[0]
		return [o_t,s_t]

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tm1
	return s_t

class DMN(object):
	def __init__(self,hidden_dim,vocab_size,batch_size = 20,bptt_truncate=-1,tm=5):
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.bptt_truncate = bptt_truncate
		self.tm = tm
		self.batch_size = batch_size
		#self.gru_p = GRUblock(hidden_dim,1)
		self.U0_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.U1_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.U2_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W0_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_i = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		

		#self.gru_em = GRUblock(hidden_dim,1)	
		self.U0_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.U1_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.U2_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W0_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W1_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.W2_em = normal_param(std=0.1, shape=(self.hidden_dim, self.hidden_dim))
		self.b0_em = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_em = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_em = constant_param(value=0.0, shape=(self.hidden_dim,))
			
		self.W1 = normal_param(std=0.1, shape=(hidden_dim, (7*hidden_dim)+2))
		self.W2 = normal_param(std=0.1, shape=(1,hidden_dim))
		self.b1 = constant_param(value=0.0, shape=(hidden_dim,))
		self.b2 = constant_param(value=0.0, shape=(1,))
		self.Wa = normal_param(std=0.1, shape=(vocab_size,hidden_dim))
		self.Wb = normal_param(std=0.1, shape=(hidden_dim,hidden_dim))
		self.U_ans0 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.U_ans1 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.U_ans2 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim+self.vocab_size))
		self.W_ans0 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim))
		self.W_ans1 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim))
		self.W_ans2 = normal_param(std=0.1, shape=(self.hidden_dim,self.hidden_dim))
		self.b_ans0 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b_ans1 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b_ans2 = constant_param(value=0.0, shape=(self.hidden_dim,))

		p = T.matrix()
		q = T.matrix()
		marked = T.ivector()
		y = T.iscalar()
		s_p,updates = theano.scan(self.input_next_state,sequences=p,		
						    outputs_info=T.zeros_like(self.b2_i))
		s_q,updates = theano.scan(self.input_next_state,sequences=q,
							outputs_info=T.zeros_like(self.b2_i))
		c = s_p.take(marked,axis=0)
		q_q = s_q[-1]
		memory = [q_q.copy()]
		for iter in range(1,self.tm+1):
			current_episode = self.new_episode(c,memory[iter-1],q_q)
			app = self.em_next_state(current_episode,memory[iter-1])
			memory.append(app)
		last_mem = memory[-1] 
		if answer_mode == 'feedforward':
			self.prediction = softmax(T.dot(self.Wa, last_mem))
		elif answer_mode == 'recurrent':
			def answer_step(a_tm1,pred_tm1):
				conc = T.concatenate([pred_tm1,q_q])
				z_t = T.nnet.hard_sigmoid(self.U_ans0.dot(conc) + self.W_ans0.dot(a_tm1) + self.b_ans0)
				r_t = T.nnet.hard_sigmoid(self.U_ans1.dot(conc) + self.W_ans1.dot(a_tm1) + self.b_ans1)
	    			h_t = T.tanh(self.U_ans2.dot(conc) + self.W_ans2.dot(a_tm1 * r_t) + self.b_ans2)
	    			s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * a_tm1
				pred = softmax(T.dot(self.Wa,s_t))
				return [s_t,pred]
			dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=theano.config.floatX))
		    	results, updates = theano.scan(fn=answer_step,outputs_info=[last_mem, T.zeros_like(dummy)],n_steps=answer_tm)
		    	self.prediction = results[1][-1]
		else:
		    raise Exception("invalid answer_module")			
		self.loss_ce = T.nnet.categorical_crossentropy(self.prediction.dimshuffle('x', 0), T.stack([y]))[0] 
		#another scan for batch training : sequences = data , loss gets added each step
		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i,
				self.U0_em,self.W0_em,self.b0_em,
				self.U1_em,self.W1_em,self.b1_em,
				self.U2_em,self.W2_em,self.b2_em,
				self.W1,self.W2,self.b1,self.b2,self.Wa,self.Wb]
		if answer_mode == 'recurrent':
			self.params = self.params + [self.U_ans0,self.W_ans0,self.b_ans0,
							self.U_ans1,self.W_ans1,self.b_ans1,
							self.U_ans2,self.W_ans2,self.b_ans2]
		#loss_ce = loss_ce + l2_reg(self.params)
		updts = upd.adadelta(self.loss_ce,self.params)
		self.train_fn = theano.function(inputs=[p,q,marked,y],outputs=[self.prediction,self.loss_ce],updates = updts)
		self.f = theano.function([p,q,marked,y],[self.loss_ce,self.prediction])
		#def save_state():

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
		g = T.nnet.sigmoid(l_2)[0]
		return g

	def new_episode_step(self,c_t,g,h_tm1):
		gru = gru_next_state(c_t,h_tm1,self.U0_em,self.W0_em,self.b0_em,self.U1_em,self.W1_em,self.b1_em,self.U2_em,self.W2_em,
					self.b2_em)
		h_t = g * gru + (1 - g) * h_tm1
		return h_t

	def new_episode(self,c,mem,q):
		g, g_updates = theano.scan(fn=self.new_attn_step,
		    sequences=c,
		    non_sequences=[mem,q],
		    outputs_info=T.zeros_like(c[0][0])) 
		e, e_updates = theano.scan(fn=self.new_episode_step,
		    sequences=[c, g],
		    outputs_info=T.zeros_like(c[0]))
		return e[-1]
	
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

	def train(self,tr_input,tr_q,tr_ans,tr_mask):
		l = len(tr_input)
		print "starting..."
		for j in range(0,50):
			a_loss = 0.0
			tr_input,tr_q,tr_ans,tr_mask = shuffle(tr_input,tr_q,tr_ans,tr_mask)
			for i in range(0,l):
				pred,loss = self.train_fn(tr_input[i],tr_q[i],tr_mask[i],tr_ans[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f"%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'states'+task+'/DMN_1.epoch%d' %(j)
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
			#print "loss : %.3f  average_loss : %.3f"%(loss,a_loss/(i+1))		
			y_true.append(test_ans[i])
			y_pred.append(pred.argmax(axis=0))
		accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
		print "accuracy: %.2f percent" % (accuracy * 100.0 / l)

def shuffle(train_input,train_q,train_answer,train_input_mask):
        print "==> Shuffling the train set"
        combined = zip(train_input,train_q,train_answer,train_input_mask)
        random.shuffle(combined)
        train_input, train_q, train_answer, train_input_mask = zip(*combined)
	return train_input,train_q,train_answer,train_input_mask
	
#parsing babi data
"""
babi_train_raw, babi_test_raw = utils.get_babi_raw("1","1")
train_input, train_q, train_answer, train_input_mask = _process_input(babi_train_raw)
a = train_input[1]
print len(train_input)
print len(a)
print len(a[0])
b = train_answer[0]
print len(train_answer)
print b
"""
#using DMN

babi_train_raw, babi_test_raw = utils.get_babi_raw(task,task)
train_input, train_q, train_answer, train_input_mask = _process_input(babi_train_raw)
test_input, test_q, test_answer, test_input_mask = _process_input(babi_test_raw)
vocab_size = len(vocab)

a1 = train_input
a2 = train_q
a3 = train_input_mask
a4 = train_answer	
dmn = DMN(word_vector_size,vocab_size)
#dmn.train(a1,a2,a4,a3)
dmn.load_state('states/state.epoch20')
dmn.test(a1,a2,a4,a3)
dmn.test(test_input,test_q,test_answer,test_input_mask)
#using GRUblock
"""
x = T.ivector()
bptt_truncate = -1
dim = 8
gru1 = GRUblock(dim)
[o,s],updates = theano.scan(gru1.calc_next_state,sequences = x,truncate_gradient=bptt_truncate,
       			    outputs_info=[None,dict(initial=T.zeros(dim))])
f = theano.function([x],s,allow_input_downcast=True)
X = np.array([0,3,4,2,0])
a = f(X)
print a[2]
"""
#using Layer
"""
x = T.ivector()
U = np.matrix([[1,2,3],[4,5,6]])
b = np.array([1,1])
l1 = Layer(U,b)
y = l1.output(x)
f = theano.function([x],y)
a = f([1,1,1])	
print a	
"""



		
