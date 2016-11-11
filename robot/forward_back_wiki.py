import numpy as np

states = ['hot', 'cold']
observations = ['hot', 'cold', 'hot']
A = np.matrix([[.25,.75,0], [0,.25,.75], [0,0,1]])
# print(A)
B = np.matrix([[1,0],[0,1],[1,0]]) #hot cold
# print(B)
pi_0 = np.matrix([[1/3],[1/3],[1/3]])
# print(pi_0)

def normalize(x):
        normalization_constant = sum(x)
        for i in range(len(x)):
          # print(x[[i]])
          x[[i]] /= normalization_constant
        return x

def fwd_bkw(observations):
  state_map = {state:i for i,state in enumerate(states)}
  # print(state_map)
  num_time_steps = len(observations)
  fmess = [None] * num_time_steps
  fmess[0] = pi_0
  ## Compute the forward messages
  for i,x_i in enumerate(observations):
    obs_index = state_map[x_i]
    # print(B[:,obs_index])
    # print()
    weights = np.multiply(fmess[i], B[:,obs_index])
    # print(weights[0])
    x = sum([A[j,:].T*w_i for j,w_i in enumerate(weights)])
    if i+1 < len(fmess):
      fmess[i+1] = normalize(x)

  # print(fmess)




  bmess = [None] * num_time_steps
  bmess[-1] = pi_0
  ## Backwards messages
  for i,x_i in enumerate(reversed(observations)):
    # print(bmess[-1-i],'backmess')
    obs_index = state_map[x_i]
    # print(B[:,obs_index])

    weights = np.multiply(bmess[-1-i], B[:,obs_index])
    x = sum([A[:,j]*w_i for j,w_i in enumerate(weights)])
    # print(x,'x\n')
    if i+1 < len(bmess):
      bmess[-2-i] = normalize(x)
    # break
  print(bmess)





  ## marginals
  # marginals = [None] * num_time_steps 
  # for i,x_i in enumerate(observations):
  #   obs_index = state_map[x_i]
  #   if i == 0:
  #     marginals[i] = normalize(np.multiply(bmess[i],  B[:,obs_index]))
  #   elif i == len(observations)-1:
  #     marginals[i] = normalize(np.multiply(fmess[i], B[:,obs_index]))
  #   else:
  #     marginals[i] = normalize(np.multiply(np.multiply(bmess[i],fmess[i]), B[:,obs_index]))
  #   print(marginals[i])
  #   print()
  # print(marginals[0:1])

def example():
    return fwd_bkw(observations)

print(example())
# for line in example():
#     print(*line)


















########################
# dictionary mess from project
########################
 # print()
    # print(marginals)
    # print(len(A))
    # x = robot.Distribution()
    # # # # forward part of the algorithm
    # for i, message in enumerate(forward_messages):
    #     prev = Distribution()
    #     x = Distribution()
    #     ## calculating the previous messages (forward_messages)
    #     ## creating weights with the observation values and model from B
    #     for state,value in message.items():
    #         obs = B[state][observations[i]]
    #         if obs == 0:
    #             pass
    #             # prev[state] = value*0
    #         else:
    #             prev[state] = obs*value
    #     ## update with transition matrix A
    #     for state,value in prev.items():
    #         if value > 0:
    #             for st,v in A[state].items():
    #                 # print(prev[state])
    #                 if x[st]:
    #                     x[st] += prev[state]*v
    #                 else:
    #                     x[st] = prev[state]*v

    #     if i < len(forward_messages)-1:
    #         forward_messages[i+1] = x.renormalize()
    #     # break
    # # print('\n--Forward Messages--')
    # # print(forward_messages[3])
    # # print('------------\n')

    # backward_messages = [None] * num_time_steps
    # num_hidden_states = len(all_possible_hidden_states)
    # backward_messages[0] = prior_distribution#{state: 1/num_hidden_states
    #                                 # for state in all_possible_hidden_states}
    # # TODO: Compute the backward messages
    # ## transpose of A[i][j] is A[j][i]
    # A_T = Distribution()

    # for state in A:
    #     for trans_state in A[state]:
    #         # print(trans_state)
    #         if not A_T[trans_state]:
    #             A_T[trans_state] = Distribution()
    #         A_T[trans_state][state] = A[state][trans_state]
    #         # print(A_T[trans_state], trans_state)
    #     # break
    # # for i in A:
    # #     for j in A[i]:
    # #         print(A[i][j] == A_T[j][i])

    #         # print(j)
    #         # break
    # # A_T = {v:A[v] for k,val in A.items() for v in val}
    # #         # print(A[i], j, i)
    # #         # print(A[j], i)
    # #         # print(j)
    # # print(A_T[state], state)
    # # print('--', A[state])
    # # x = robot.Distribution()