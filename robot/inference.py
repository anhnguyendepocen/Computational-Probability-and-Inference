#!/usr/bin/env python
# inference.py
import collections
import sys

import graphics
import numpy as np
import robot
from robot import Distribution


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

trans_state_index_map = {s:i for i,s in enumerate(all_possible_hidden_states)}
obs_state_index_map = {o:i for i,o in enumerate(all_possible_observed_states)}


A = np.array([[float(0)]*len(all_possible_hidden_states)]\
                        *len(all_possible_hidden_states))
B = np.array([[float(0)]*len(all_possible_observed_states)]\
                        *len(all_possible_hidden_states))

for i, state in enumerate(all_possible_hidden_states):
    trans_model = transition_model(state)

    for ind,v in trans_model.items():
        map_i = trans_state_index_map[ind]
        A[i,map_i] = v
    ob_model = observation_model(state)
    for ob, v in ob_model.items():
        obs_i = obs_state_index_map[ob]
        B[i,obs_i] = v

# np.set_printoptions(precision=3, linewidth=200) 
# print(A[:10, :10], '\n') 
# print(B[:10, :10], '\n')

prior_matrix = np.array([float(0)]*len(all_possible_hidden_states)) 
for st,value in prior_distribution.items():
    state_idx = trans_state_index_map[st]
    prior_matrix[state_idx] = value

# print(prior_matrix[2], '\n')
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log(base 2) of a non-negative real number
    a = np.array(x)
    for i,row in enumerate(x): #row
        for col,v in enumerate(row):
            if v == 0:
                a[i][col] = np.inf
            else:
                a[i][col] = -np.log2(v)
    return a

# def normalize(x):
#     # print()
#     normalization_constant = sum(x)
#     for i in range(len(x)):
#     # print(x[[i]])
#         x[[i]] /= normalization_constant
#     return x

# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Summary
    -----
    Given observations, we get an idea of which state the robot is currently in,
    so given the possibles stated returned by observation model, we can find the
    message for the potential values for the next step using the tansition
    matrix.

    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    #
    # observations = [(4, 3), (4, 2), (3, 2), (4, 0), (2, 0), (2, 0), (3, 2), 
    #                 (4, 2), (2, 3), (3, 5)]
    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_matrix
    # # # TODO: Compute the forward messages
    for i,x_i in enumerate(observations):

        if x_i:
            obs_index = obs_state_index_map[x_i]
            pi_0 = forward_messages[i]
            weights = np.multiply(pi_0, B[:,obs_index])
        else:
            weights = forward_messages[i]

        x = sum([A[j,:]* w_i.T for j,w_i in enumerate(weights)])

        if i+1 < len(forward_messages):
            forward_messages[i+1] = x#normalize(x)
        # break



    backward_messages = [None] * num_time_steps
    message = np.ones(len(all_possible_hidden_states), dtype=np.float64)
    backward_messages[-1] = message/len(all_possible_hidden_states)
   
# ****
    ## Backwards messages
    for i,x_i in enumerate(reversed(observations)):
        if x_i:
            obs_index = obs_state_index_map[x_i]
            pi = backward_messages[-1-i]
            weights = np.multiply(pi, B[:,obs_index])
        else:
            weights = backward_messages[-1-i]
        x = sum([A[:,j]*w_i for j,w_i in enumerate(weights)])

        if i+1 < len(backward_messages):
            backward_messages[-2-i] = x#normalize(x)

  ## marginals as matrix
    marginals = [None] * num_time_steps 
    for i,x_i in enumerate(observations):
        if x_i:
            obs_index = obs_state_index_map[x_i]
            marginals[i] = np.multiply(np.multiply(backward_messages[i],
                                                   forward_messages[i]),
                                       B[:,obs_index])
        else:
            marginals[i] = np.multiply(backward_messages[i],forward_messages[i])


    ## marginals as dictionary
    marg_dict = [None]*num_time_steps
    for j,m in enumerate(marginals):
        x = Distribution()
        for i,x_i in enumerate(m):
            if x_i == 0 or x_i==None:
                continue
            x[all_possible_hidden_states[i]] = x_i

        marg_dict[j] = x.renormalize()

    return marg_dict


def Viterbi(observations, second_best=False):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # observations = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    # observations = [(1, 6), (4, 6), (4, 7), None, (5, 6), (6, 5), (6, 6), None, (5, 5), (4, 4)]
    # observations = [(1, 4), (1, 5), (1, 5), (1, 6), (0, 7), (1, 7), (3, 7), (4, 7), (4, 7), (4, 7)]#2 second best
    log_B = careful_log(B)# log2 for each obseravtion distribution
    log_A = careful_log(A)# log2 for each transition distribution

    num_time_steps = len(observations)
    mes = [None]*num_time_steps
    traceback = [None]*num_time_steps

    # logs of prior, initial message
    mes[0] = np.array([-np.log2(i) if i != 0 else np.inf for i in prior_matrix])

    for i, x_i in enumerate(observations):
        if x_i:
            obs_i = obs_state_index_map[x_i]
            phi = log_B[:, obs_i] # current observation distribution
        prev = mes[i]
        m = {} # current message holder
        t = {} # current traceback message holder
        if i == 0 and x_i:
            for s_i, state in enumerate(all_possible_hidden_states):
                cur = log_A[:, s_i] + phi + mes[0] 
                t[state] = all_possible_hidden_states[np.argmin(cur)]
                m[state] = cur.min()
        elif x_i is None: # when no obs, don't include obs in calculation
            prev = np.array([mes[i-1][state] 
                            for state in all_possible_hidden_states])
            for s_i, state in enumerate(all_possible_hidden_states):
                cur = log_A[:,s_i] + prev
                t[state] = all_possible_hidden_states[np.argmin(cur)]
                m[state] = cur.min()
        else:
            prev = np.array([mes[i-1][state] 
                            for state in all_possible_hidden_states])
            for s_i, state in enumerate(all_possible_hidden_states):
                cur = log_A[:,s_i] + phi + prev
                t[state] = all_possible_hidden_states[np.argmin(cur)]
                m[state] = cur.min()

        if i < len(observations):
            # if i == 1: print(m)
            mes[i] = m
            traceback[i] = t

    last_ob_index = obs_state_index_map[observations[-1]]
    last_message = np.array([mes[-1][state] 
                            for state in all_possible_hidden_states])
    x_hat = all_possible_hidden_states[np.argmin(last_message + 
                                                 log_B[:,last_ob_index])]

    tracer = [0]*len(observations)
    tracer[0] = x_hat
    l = [] # viterbi holder

    for i,tr in enumerate(reversed(traceback)):
        l.append(tr[tracer[i]])
        if i == len(traceback)-1: break
        tracer[i+1] = tr[tracer[i]]

    viterbi = l[::-1]

    return viterbi

def careful_log2(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """
    # commented out is attempt using matrices :/ still working on it
    # working version using dictionaries
    # -------------------------------------------------------------------------
    # observations = [(8, 2), (8, 1), (10, 0), (10, 0), (10, 1), (11, 0), (11, 0), (11, 1), (11, 2), (11, 2)]
    # **********************observations = [(1, 4), (1, 5), (1, 5), (1, 6), (0, 7), (1, 7), (3, 7), (4, 7), (4, 7), (4, 7)]#2   

    # log_B = careful_log(B)#((list(map(careful_log, B)))
    # log_A = careful_log(A)

    # num_time_steps = len(observations)
    # mes = [None]*num_time_steps
    # traceback = [None]*num_time_steps
    # mes2 = [None]*num_time_steps
    # traceback2 = [None]*num_time_steps

    # mes[0] = np.array([-np.log2(i) if i != 0 else np.inf 
    #                   for i in prior_matrix])
    # mes2[0] = np.array([-np.log2(i) if i != 0 else np.inf 
    #                    for i in prior_matrix])
    # for i, x_i in enumerate(observations):
    #     if x_i:
    #         obs_i = obs_state_index_map[x_i]
    #         phi = log_B[:, obs_i]
    #     # prev = mes[i]
    #     # prev2 = mes2[i]
    #     m = {}
    #     t = {}
    #     m2 = {}
    #     t2 = {}
    #     if i == 0 and x_i:
    #         for s_i, state in enumerate(all_possible_hidden_states):
    #             cur = log_A[:, s_i] + phi + mes[0] 
    #             t[state] = all_possible_hidden_states[np.argmin(cur)]
    #             m[state] = cur.min()
    #             ## second best
    #             cur[np.argmin(cur)] = np.inf
    #             if cur[cur.argmin()] == np.inf:
    #                 t2[state] = t[state]
    #                 m2[state] = m[state]
    #             else:
    #                 t2[state] = all_possible_hidden_states[np.argmin(cur)]
    #                 m2[state] = cur.min()

    #     elif x_i is None:
    #         prev = np.array([mes[i-1][state] 
    #                         for state in all_possible_hidden_states])
    #         prev2 = np.array([mes2[i-1][state]
    #                          for state in all_possible_hidden_states])
    #         # print(prev2)
    #         for s_i, state in enumerate(all_possible_hidden_states):
    #             cur = log_A[:,s_i] + prev
    #             t[state] = all_possible_hidden_states[np.argmin(cur)]
    #             m[state] = cur.min()
    #             ## second best
    #             cur[np.argmin(cur)] = np.inf
    #             if cur[cur.argmin()] == np.inf:
    #                 t2[state] = t[state]
    #                 m2[state] = m[state]
    #                 # print(1)
    #             else:
    #                 t2[state] = all_possible_hidden_states[np.argmin(cur)]
    #                 m2[state] = cur.min()

    #     else:
    #         # print(mes2[1])
    #        prev = np.array([mes[i-1][state]
    #                        for state in all_possible_hidden_states])
    #        prev2 = np.array([mes2[i-1][state] 
    #                          for state in all_possible_hidden_states])
    #         for s_i, state in enumerate(all_possible_hidden_states):
    #             cur = log_A[:,s_i] + phi + prev
    #             a = np.argmin(cur)
    #             t[state] = all_possible_hidden_states[np.argmin(cur)]
    #             m[state] = cur.min()
    #             ## secondbest
    #             cur[np.argmin(cur)] = np.inf
    #             if cur[cur.argmin()] == np.inf:
    #                 t2[state] = t[state]
    #                 m2[state] = m[state]
    #                 # print(1)
    #             else:
    #                 t2[state] = all_possible_hidden_states[np.argmin(cur)]
    #                 m2[state] = cur.min()
    #     # x = None
    #     # if i == 1:
    #     #     x = {state: m[state] for state in all_possible_hidden_states}
    #     #     x2 = {state: m2[state] for state in all_possible_hidden_states}
    #     #     a = {k:v for k,v in x.items() if v != np.inf}
    #     #     a2 = {k:v for k,v in x2.items() if v != np.inf}
    #     # if x: print(a, '\n\n', a2)#, a[(1, 0, 'stay')])



    #     if i < len(observations):
    #         # if i == 1: print(t2)
    #         mes[i] = m
    #         traceback[i] = t#[i for i in t]
    #         mes2[i] = m2
    #         traceback2[i] = t2
    #     # if i == 2:
    #     #     break

    # last_ob_index = obs_state_index_map[observations[-1]]
    # last_message = np.array([mes2[-1][state] 
    #                         for state in all_possible_hidden_states])
    # # # last_message[]
    # x_hat = all_possible_hidden_states[np.argmin(last_message + 
    #                                              log_B[:, last_ob_index])]


    # # print(x_hat)
    # tracer = [0]*len(observations)
    # tracer[0] = x_hat

    # l = []

    # # print(traceback2)
    # for i,tr in enumerate(reversed(traceback2)):
    #     # print(tr[tracer[i]], traceback[i][tracer[i]])
    #     # break
    #     # if tr[tracer[i]] < traceback[i][tracer[i]]:
    #     l.append(tr[tracer[i]])
    #     # else:
    #         # l.append(traceback[i][tracer[i]])
    #     if i == len(traceback2)-1: break
    #     tracer[i+1] = tr[tracer[i]]

    # second_best = l[::-1]

    # print(second_best)

    # # print('\nExpected: ', [[9, 2, "stay"], [9, 1, "up"], [9, 0, "up"], [9, 0, "stay"], [10, 0, "right"], [11, 0, "right"], [11, 0, "stay"], [11, 0, "stay"], [11, 1, "down"], [11, 2, "down"]])
    # print('\n(2) Expected: ', [[1, 4, "stay"], [1, 5, "down"], [1, 6, "down"], [1, 7, "down"], [1, 7, "stay"], [1, 7, "stay"], [2, 7, "right"], [3, 7, "right"], [4, 7, "right"], [5, 7, "right"]])
    # # num_time_steps = len(observations)
    # # estimated_hidden_states = [None] * num_time_steps # remove this

    # return second_best#estimated_hidden_states***********************
    num_time_steps = len(observations)

    # Basically for each (possible) hidden state at time step i, we need to
    # keep track of the best previous hidden state AND the second best
    # previous hidden state--where we need to keep track of TWO back pointers
    # per (possible) hidden state at each time step!

    messages = []  # best values so far
    messages2 = []  # second-best values so far
    back_pointers = []  # per time step per hidden state, we now need
    # *two* back-pointers

    # -------------------------------------------------------------------------
    # Fold observations into singleton potentials
    #
    phis = []  # phis[n] is the singleton potential for node n
    for n in range(num_time_steps):
        potential = robot.Distribution()
        observed_state = observations[n]
        if n == 0:
            for hidden_state in prior_distribution:
                value = prior_distribution[hidden_state]
                if observed_state is not None:
                    value *= observation_model(hidden_state)[observed_state]
                if value > 0:  # only store entries with nonzero prob.
                    potential[hidden_state] = value
        else:
            for hidden_state in all_possible_hidden_states:
                if observed_state is None:
                    # singleton potential should be identically 1
                    potential[hidden_state] = 1.
                else:
                    value = observation_model(hidden_state)[observed_state]
                    if value > 0:  # only store entries with nonzero prob.
                        potential[hidden_state] = value
        phis.append(potential)

    # -------------------------------------------------------------------------
    # Forward pass
    #

    # handle initial time step differently
    initial_message = {}
    for hidden_state in prior_distribution:
        value = -careful_log2(phis[0][hidden_state])
        if value < np.inf:  # only store entries with nonzero prob.
            initial_message[hidden_state] = value
    messages.append(initial_message)
    initial_message2 = {}  # there is no second-best option
    messages2.append(initial_message2)

    # rest of the time steps
    for n in range(1, num_time_steps):
        prev_message = messages[-1]
        prev_message2 = messages2[-1]
        new_message = {}
        new_message2 = {}
        new_back_pointers = {}  # need to store 2 per possible hidden state

        # only look at possible hidden states given observation
        for hidden_state in phis[n]:
            values = []
            # each entry in values will be a tuple of the form:
            # (<value>, <previous hidden state>,
            #  <which back pointer we followed>),
            # where <which back pointer we followed> is 0 (best back pointer)
            # or 1 (second-best back pointer)

            # iterate through best previous values
            for prev_hidden_state in prev_message:
                value = prev_message[prev_hidden_state] - \
                    careful_log2(transition_model(prev_hidden_state)[
                        hidden_state]) - \
                    careful_log2(phis[n][hidden_state])
                if value < np.inf:
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 0))

            # also iterate through second-best previous values
            for prev_hidden_state in prev_message2:
                value = prev_message2[prev_hidden_state] - \
                    careful_log2(transition_model(prev_hidden_state)[
                        hidden_state]) - \
                    careful_log2(phis[n][hidden_state])
                if value < np.inf:
                    # only store entries with nonzero prob.
                    values.append((value, prev_hidden_state, 1))

            if len(values) > 0:
                # this part could actually be sped up by not using a sorting
                # algorithm...
                sorted_values = sorted(values, key=lambda x: x[0])
                best_value, best_prev_hidden_state, which_back_pointer = \
                    sorted_values[0]

                # for the best value, the back pointer should *always* be 0,
                # meaning that we follow the best back pointer and not the
                # second best
                if len(values) > 1:
                    best_value2, best_prev_hidden_state2, which_back_pointer2\
                        = sorted_values[1]
                else:
                    best_value2 = np.inf
                    best_prev_hidden_state2 = None
                    which_back_pointer2 = None

                new_message[hidden_state] = best_value
                new_message2[hidden_state] = best_value2
                new_back_pointers[hidden_state] = \
                    ((best_prev_hidden_state, which_back_pointer),
                     (best_prev_hidden_state2, which_back_pointer2))

        messages.append(new_message)
        messages2.append(new_message2)
        back_pointers.append(new_back_pointers)

    # -------------------------------------------------------------------------
    # Backward pass (follow back-pointers)
    #

    # handle last time step differently
    values = []
    for hidden_state, value in messages[-1].items():
        values.append((value, hidden_state, 0))
    for hidden_state, value in messages2[-1].items():
        values.append((value, hidden_state, 1))

    divergence_time_step = -1

    if len(values) > 1:
        # this part could actually be sped up by not using a sorting
        # algorithm...
        sorted_values = sorted(values, key=lambda x: x[0])

        second_best_value, hidden_state, which_back_pointer = sorted_values[1]
        estimated_hidden_states = [hidden_state]

        # rest of the time steps
        for t in range(num_time_steps - 2, -1, -1):
            hidden_state, which_back_pointer = \
                back_pointers[t][hidden_state][which_back_pointer]
            estimated_hidden_states.insert(0, hidden_state)
    else:
        # this happens if there isn't a second best option, which should mean
        # that the only possible option (the MAP estimate) is the only
        # solution with 0 error
        estimated_hidden_states = [None] * num_time_steps

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)
    
    # observations = [(2, 0), (2, 0), (3, 0), (4, 0), (4, 0), (6, 0), (6, 1), (5, 0), (6, 0), (6, 2)]
    # num_time_steps = len(observations)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
