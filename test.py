import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import pydot
import cPickle as pickle
import os as os
import utils
import init

hidden_dim = 6

def constant_param(value=0.0, shape=(0,)):
    return theano.shared(init.Constant(value).sample(shape), borrow=True)
   
def normal_param(std=0.1, mean=0.0, shape=(0,)):
    return theano.shared(init.Normal(std, mean).sample(shape), borrow=True)

U0_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
U1_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
U2_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
W0_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
W1_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
W2_em = normal_param(std=0.1, shape=(hidden_dim, hidden_dim))
b0_em = constant_param(value=0.0, shape=(hidden_dim,))
b1_em = constant_param(value=0.0, shape=(hidden_dim,))
b2_em = constant_param(value=0.0, shape=(hidden_dim,))
			
W1 = normal_param(std=0.1, shape=(hidden_dim, (7*hidden_dim)+2))
W2 = normal_param(std=0.1, shape=(1,hidden_dim))
b1 = constant_param(value=0.0, shape=(hidden_dim,))
b2 = constant_param(value=0.0, shape=(1,))
Wb = normal_param(std=0.1, shape=(hidden_dim,hidden_dim))

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = (T.ones_like(z_t) - z_t) * h_t + z_t * s_tm1
	return s_t

def new_attn_step(c_t,g_tm1,m_im1,q):
	cWq = T.stack([T.dot(T.dot(c_t, Wb), q)])
       	cWm = T.stack([T.dot(T.dot(c_t, Wb), m_im1)])
	z = T.concatenate([c_t,m_im1,q,c_t*q,c_t*m_im1,T.abs_(c_t-q),T.abs_(c_t-m_im1),cWq,cWm],axis=0)
	l_1 = T.dot(W1, z) + b1
	l_1 = T.tanh(l_1)
	l_2 = T.dot(W2,l_1) + b2
	g = T.nnet.sigmoid(l_2)[0]
	return g

def new_episode_step(c_t,g,h_tm1):
	gru = gru_next_state(c_t,h_tm1,U0_em,W0_em,b0_em,U1_em,W1_em,b1_em,U2_em,W2_em,b2_em)
	h_t = g * gru + (1 - g) * h_tm1
	return h_t

def new_episode(c,mem,q):
	g, g_updates = theano.scan(fn=new_attn_step,sequences=c,non_sequences=[mem,q],outputs_info=T.zeros_like(c[0][0])) 

	e, e_updates = theano.scan(fn=new_episode_step,sequences=[c, g],outputs_info=T.zeros_like(c[0]))

	gs = T.nnet.softmax(g)[0]
	return gs,e[-1]
c_t = T.matrix()
q_q = T.vector()
m = T.matrix()
#G,c_e= new_episode(c,q_q.copy(),q_q)
def step(m_tm1,c,q):
	G,c_e= new_episode(c,m_tm1,q)
	m_t = m_tm1 + c_e
	return G,m_t
epm,epm_updates = theano.scan(fn = step,non_sequences=[c_t,q_q],outputs_info = [None,q_q.copy()],n_steps = 3)
f = theano.function([c_t,q_q],epm[0])
g = theano.function([c_t,q_q],epm[1])
"""
def step_em(c,q,m_tm1):
	G,current_episode = new_episode(c,m_tm1,q)
	m_t = em_next_state(current_episode,m_tm1)
	return G,m_t

c = T.matrix()
q_q = T.vector()
epm,epm_updates = theano.scan(fn = step_em,non_sequences=[c,q_q], outputs_info=[None,T.zeros_like(q_q)],n_steps=3)
f = theano.function([c,q_q],epm[0])
g = theano.function([c,q_q],epm[1])"""

a = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.044457],
	[0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
	[0.33042, 0.24995, -0.60874, 0.10923, 0.036372, 0.151],
	[0.088359, 0.16351, -0.21634, -0.094375, 0.018324, 0.21048],
	[-0.43478, -0.31086, -0.44999, -0.29486, 0.16608, 0.11963]]
b = [-0.16899, 0.40951, 0.63812, 0.47709, -0.42852, -0.55641]

c = f(a,b)
d = g(a,b)
print c
print "****************************"
print d










