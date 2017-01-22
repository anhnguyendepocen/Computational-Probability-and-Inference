import numpy as np

observations = 'H H T T T'.split()
states = ['fair', 'biased']
obs_states = ['H', 'T']
states_map = {s:i for i,s in enumerate(states)}
obs_map = {s:i for i,s in enumerate(obs_states)}
A = np.array([[3/4, 1/4], [1/4, 3/4]]) #(0,0) --> fair,fair; (0,1)-->fair,biased
# recall B matrix is observation state (HHTTT)
# first row is 1/2*1/2=1/4, 1/2*1/4=1/8 fair/biased
B = np.array([[1/4, 1/8], [1/2, 1/4], [1/2, 3/4], [1/2, 3/4], [1/2, 3/4]]) #row-fair/biased , col-h or t P(Yi|Xi)
# print(B)
# print(A)
mes = [None]*(len(observations)-1)
traceback = [None]*(len(observations)-1)

for i, x_i in enumerate(observations):
	if i == len(observations)-1: break
	obs_i = obs_map[x_i]
	phi = -np.log2(B[i,:]) ## hmmm
	# print(phi)

	m={} #messages
	t={} #taceback messages
	if i == 0:
		# phi = -np.log2(B[:,i].T)
		for s_i, state in enumerate(states):
			cur = -np.log2(A[s_i,:])+phi
			t[state] = states[np.argmin(cur)]
			m[state] = cur.min()
	else:
		prev = [mes[i-1][state] for state in states]

		for s_i, state in enumerate(states):
			cur = -np.log2(A[s_i,:]) + phi + prev
			t[state] = states[np.argmin(cur)]
			m[state] = cur.min()

	if i < len(observations)-1:
		mes[i] = m
		traceback[i] = t
### traceback - get viterbi
x_hat = states[np.argmin(np.array([mes[-1][state] for state in states]) - np.log2(B[-1,:]))]

tracer = [0]*len(observations)
tracer[0] = x_hat
# print(tracer)
for i,tr in enumerate(reversed(traceback)):
	# print(tr['biased'])
	tracer[i+1] = tr[tracer[i]]
viterbi = tracer[::-1]
print(viterbi)

	# mapper = str(tracer[-1-i])
	# print(tr[mapper])
	# else:

	# print()
	# prev = 
	# print(i[])


# 
# print(x_hat)
# print(mes)
# print(traceback)
# min_ = -np.inf
# for i in reversed(mes):
# 	print(min(i))

	# break
	# for s in A:
	# 	print(s)
	# m = {s:phi[s_i2] - np.log2(A[s_i, s_i2]) for s_i, s in enumerate(states)
											# for s_i2 in range(len(states))}
	# print(m)
			# print(-np.log2(A[s2_i, s_ind]), s_ind, s2_i)
			# print(state,s)
			# print(s_i, s_i2)
			# print()
			# print(-np.log2(A[s_i, s_i2]))
			# print()
			
		# print(phi)
# print(mes)
	# 	break
	# break
	# : ## previous states in projy
	# mes[i] = [-np.log2(B[i, obs_i]) - np.log2(A[obs_i, state_i]) for state, state_i in states_map.items()]
	# print(phi)
		# cur = min([phi + np.log(A)
# print(mes)
# def 