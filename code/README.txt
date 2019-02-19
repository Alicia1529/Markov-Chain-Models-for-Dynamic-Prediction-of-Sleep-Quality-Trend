1. Data file£º alicia-data.txt; anqi-data.txt;skye-data.txt
2. data.py: Transform the original curve into five states and plot the graph
3. dice.py: imported by MC.py, to generate new figure in terms of the probability
4. MarkovChianModel.py: first order, second order and third order Markov Chain model. Compute the transition matrix, do dependency test, temporary stationary test, make predictions of the sleeping curve of the whole night or a given time interval. Compute the probability of need to wake up as well as the error rate in the end.
5. MixtureTrnasitionDistributionModel.py: Use numerical maximization of log-likelihood, developed by Berchtold, to estimated parameters in MTD model
6. EM-MTDg.py: Use EM algorithm to maximize the log-likehood of MTDg model.
