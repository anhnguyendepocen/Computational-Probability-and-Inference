#Computational Probability and Inference
Probability projects including:
- Movie recommendations
- Robot Localization
- Spam detector
- ...<br>

Have worked examples of basic probability concepts implemented in Python--conditioning, marginalizing. As well as some information theory concepts--entorpy, informationa gain, majority vote (in Lecture folder).
<br>
###Movie Recommendation
Predict the rating of a new (unseen) movie, X, given set of ratings for movies seen.<br>
<strong>How it's done:</strong><br>
 - Given ratings for a set a movies recommends, compute the posterior distribution, find likelihood of a distibution of the given observation (movie rating). Once we have the likelihood distribution, we use inference (bayes) to get a ditstributions of posterior for each possible rating (and its likelihood) for each movie in data. In addition we provide a MAP (Maximum a posteriori) estimate for each movie, which is really what would be provided in end; distibution is interesting though. Once we have the distribution of possible ratings for each movie, implement an entropy function and get the entropy (randomness) for each of the posterior values.<br>

###Robot Localization
Get and predict robot location, given observation values from sensor: says robot in one of five locations on grid.<br>
<strong>How it's done:</strong><br>
 - Given observation from noisy sensor, try to located and estimate the robot's hidden location. The robot has a transition model of some distribution. Treating as hidden markov model (HMM) implement forward-backward algorithm, by getting distribution of current observation (phi) with the previous message (state in previous step, prior initially), then get sum of the distribution of the transition model for each of the possible values from weights (phi*previous message). This becomes the message for the next stage, iterate for all obs, getting forward messages. Similar steps for backwards, using transpose (or columns) of the transition distribution matrix. Once we have the forward and backward messages, we again loop through the observations calculating the marginals for eack observation (backward_message[i] * forward_message[i] * phi (observations distribution) ). Next we implement and calculate the MAP estimation for HMMs using Viterbi algorithm. To implement Viterbi I used Min-Sum for HMMs, where each message is the minimum of the sum of the $-log_2$ of the previous message, phi (obervation), and psi (transition), storing the argmin for traceback messages needed for viterbi. Once we get the minimum over all possible states, we get the argmin for the last last message and use our traceback to get the MAP estimate for Viterbi.<br>



