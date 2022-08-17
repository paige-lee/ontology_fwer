#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import pandas as pd
import re


# In[ ]:


def clean_adjacency_mat(adjacency_matrix):
    ''' Function that cleans the adjacency matrix 
    
    This function cleans the raw adjacency matrix so that parents come before children. 
    This converts between one convention and another convention. It doesn't check for parents vs. children.
    
    Parameters
    ----------
    adjacency_matrix: pandas.DataFrame
        A square adjacency matrix, where 1 in the (i, j) entry means the ith structure is a parent of the 
        jth structure.
    
    Returns
    ----------
    pandas.DataFrame
        A cleaned adjacency matrix (parents come before children)
        
    '''
    
    # Reverse the order of columns 
    columns = adjacency_matrix.columns.tolist()
    columns = columns[::-1]
    adjacency_matrix = adjacency_matrix[columns]
    
    # Reverse the order of rows
    adjacency_matrix = adjacency_matrix[::-1]
    
    # Take the transpose of the matrix
    adjacency_matrix = adjacency_matrix.T
    
    return adjacency_matrix


# In[ ]:


def clean_multilevel(multilevel, adjacency_matrix):
    """ Function that cleans the multilevel lookup table
    
    This function cleans the raw multilevel lookup table so that parents come before children. 
    The multilevel lookup table uses datasets from https://mricloud.org
    
    Parameters
    ----------
    multilevel: pandas.DataFrame
        multilevel lookup table, whose columns are "Structure," "Immediate.parent," and 
        "Immediate.child.children," and each row is a structure in the ontology. 
    
    adjacency_matrix: pandas.DataFrame
        A cleaned adjacency matrix (parents come before children)
    
    Returns
    ----------
    pandas.DataFrame
        Cleaned multilevel lookup table (parents come before children)
   
    """
    
    # Reverse the order of rows
    multilevel = multilevel[::-1]
    
    # Reassign the numbers
    multilevel.Number = range(1, adjacency_matrix.shape[0] + 1) 
    
    return multilevel


# In[ ]:


def subset_matrix_creator(subset_leaf_list, adjacency_matrix, multilevel):
    ''' Function that creates a subset matrix
    
    This function creates a subset of the adjacency matrix using the user-specified structures.
    
    Parameters
    ----------
    
    subset_leaf_list: list
        List of leaf structures to include in the subset
    
    adjacency_matrix: pandas.DataFrame
        The cleaned adjacency matrix
    
    multilevel: pandas.DataFrame
        The cleaned multilevel lookup table
    
    Returns
    ----------
    pandas.DataFrame
        A square subset matrix of the adjacency matrix that includes the user-specified leaf structures and 
        all of their parents
    
    '''
    
    full_subset_list = subset_leaf_list # This list will get filled
    iterations = 5 * len(subset_leaf_list) - 1
    
    for i in range(0, len(subset_leaf_list)): # For each leaf structure
        for j in range(0, iterations): # Iterate over the 4 other levels in this ontology (5 levels * number of leaves)

            # Structure index
            structure_index = int(np.where(multilevel["Structure"] == full_subset_list[j])[0])
            structure_index = (adjacency_matrix.shape[0]) - structure_index
            structure = multilevel["Structure"][structure_index]

            full_subset_list.append(multilevel["Immediate.parent"][structure_index])
        
    full_subset_list = set(full_subset_list)
    full_subset_list = pd.DataFrame(full_subset_list)
        
    pattern = r"_[0-9]+_"
    structure_numbers = []
    
    for i in range(0, full_subset_list.shape[0]):
        if full_subset_list[0][i] == "Everything":
            structure_numbers.append(float("inf")) # Assign the number "inf" to "Everything" for flexibility
        else:
            number = re.findall(pattern, full_subset_list[0][i])[0]
            number = re.sub("[^0-9]", "", number)
            number = int(number)
            structure_numbers.append(number)
    
    full_subset_list["structure_numbers"] = structure_numbers
    full_subset_list = full_subset_list.sort_values(by = ["structure_numbers"], axis = 0)
    full_subset_list = full_subset_list[0].tolist()
    
    # Index the rows and columns of the adjacency matrix by these structures to create a subset
    subset = adjacency_matrix[full_subset_list]
    subset = subset.loc[full_subset_list]

    # Reverse the order of rows and columns in the subset of the adjacency matrix
    cols = subset.columns.tolist()
    cols = cols[::-1]
    subset = subset[cols]
    subset = subset[::-1]
    
    return subset


# In[ ]:


def adjacency_descendants(adjacency_matrix, N, mu):
    ''' Function that creates an adjacency matrix of descendants
    
    Sometimes we're interesting in querying whether one structure is a descendant of the other as opposed to 
    a direct child.
    
    Parameters
    ----------
    
    adjacency_matrix: binary numpy.array
        The cleaned adjacency matrix
    
    N: int
        The number of samples
    
    mu: float
        The difference in means (generally unknown)
    
    Note: this function may only be used for an ontology that has 6 levels ("Everything" is the 
    highest/most general level).
    
    Returns
    ----------
    binary numpy.array
        Transitive adjacency matrix 
    
    '''
    
    M = adjacency_matrix.shape[0] # Total number of unique structures
    names_full = adjacency_matrix.columns # List of the 509 structures' names
    A = np.array(adjacency_matrix, dtype = bool)
    Descendants = np.copy(A)
    
    Descendants = np.logical_or(Descendants,Descendants@A)
    Descendants = np.logical_or(Descendants,Descendants@A)
    Descendants = np.logical_or(Descendants,Descendants@A)
    Descendants = np.logical_or(Descendants,Descendants@A)
    Descendants = np.logical_or(Descendants,Descendants@A)
    Descendants = np.logical_or(Descendants,Descendants@A)
    
    return Descendants


# In[ ]:


def adjacency_ancestors(adjacency_matrix, N, mu):
    ''' Function that creates an adjacency matrix of ancestors
    
    Sometimes we're interesting in querying whether one structure is an ancestor of the other as opposed to 
    a direct parent.
    
    Parameters
    ----------
    adjacency_matrix: pandas.DataFrame
        The cleaned adjacency matrix
    
    N: int
        The number of samples
    
    mu: float
        The difference in means (generally unknown)
    
    Note: this function may only be used for an ontology that has 6 levels ("Everything" is the 
    highest/most general level).
    
    Returns
    ----------
    binary numpy.array
        Transitive adjacency matrix 
        
    '''
    
    M = adjacency_matrix.shape[0]
    Descendants = adjacency_descendants(adjacency_matrix, N, mu)
    Ancestors = Descendants.T # Take transpose of descendants matrix to get ancestors    
    Ancestors_and_self = np.logical_or(Ancestors,np.eye(M))
    
    return Ancestors


# In[ ]:


def phi(x,mu=0.0):
    ''' Standard normal distribution CDF
    
    A Gaussian probability density function with unit variance
    
    Parameters
    ----------
    x: float or numpy.array
        A point to evaluate the function at, can be a numpy array
    
    mu: float
        Mean of the gaussian (default 0)
    
    Returns
    ----------
    numpy.array
        The Gaussian probability density function evaluated at specified point
    
    Note: this function is not called
    
    '''
    return 1.0/np.sqrt(2.0*np.pi)*np.exp(-(x - mu)**2/2.0)


# In[ ]:


def P_from_Q(Q,Ancestors_and_self):
    ''' Function that calculates P from Q 
    
    This function computes the marginal probability that a structure is affected given the conditional 
    probabilities that structures are affected conidtioned on their parents.
    
    Parameters
    ----------
    Q: numpy.array
        List of conditional probabilities 
        
    Ancestors_and_self: transitive adjacency matrix 
        Adjacency matrix with loops (when a node is connected to itself)
    
    Returns 
    ----------
    numpy.array
        marginal probabilities 
    
    Note: this function is not called
    
    '''
    P = np.empty_like(Q)
    for i in range(M):
        P[i] = np.prod(Q[Ancestors_and_self[i,:]])
    return P


# In[ ]:


def Q_from_P(P,A):
    ''' Function that calculates Q from P
    
    Given a list of marginal probabilities of structures being affected, compute the conditional probabilities 
    of a structure being affected given its parents.
    
    Parameters
    ----------
    P: numpy.array
        The marignal probabilities
    
    A: binary numpy.array
        Adjacency matrix that describes parent to child relationships
    
    Returns
    ----------
    numpy.array
        The conditional probabilities
    
    '''
    M = A.shape[0]
    # now we need to calculate Q
    Q = np.zeros_like(P)
    Q[0] = P[0]
    for i in range(1,M):
        Q[i] = P[i] / P[A[:,i]]
    return Q


# In[ ]:


def estimate_P(X, mu, A, Descendants_and_self, draw=False, niter=100, P0=None, names=None, clip=0.001):
    ''' Function for estimating P
    
    Apply an EM algorithm to estimate marginal probabiltiies that each structure is affected given a dataset.
    We assume data is normally distributed with unit variance and mean mu.
    
    Parameters
    ----------
    X: numpy.array
        Contains observations for each structure and each subject
    
    mu: float
        The known mean for affected structures (unaffected structures have mean 0)
    
    A: binary numpy.array
        Adjacency matrix describing parent child relationships
    
    Descendants_and_self: transitive adjacency matrix
        Adjacency matrix of descendants. Can be computed from A, but here we use it as an input
    
    draw: int
        Illustrates the data every `draw` iterations of EM algorithm. 
        draw = 0 or false means do not draw. Default value is False.
    
    iter: int
        The number of iterations of em algorithm
    
    p0: float
        The initial guess for marginal probabilities

    names: list 
        The names (strings) of structures in ontology
        
    clip: float
        Number that clips probabilities away from 0 or 1

    Returns
    ----------
    numpy.array
        Contains the marginal probabilities that structures are affected)

    '''
    
    if draw: 
        f,ax = plt.subplots(2,2)
        if names is None:
            names = np.arange(A.shape[0])

    N = X.shape[0]
    m = X.shape[1]
    M = A.shape[0]
    is_leaf = np.sum(Descendants_and_self, 1) == 1
    
    # okay now comes my algorithm
    # initialize
    if P0 is None:
        P = np.ones(M)*0.5
    else:
        P = np.asarray(P0)
    
    for it in range(niter):
        # calculate leaf posterior (this is prob of no effect)
        #leaf_posterior = ((1.0-P[is_leaf])*phi(X))
        #leaf_posterior = leaf_posterior/(leaf_posterior + P[is_leaf]*phi(X,mu) )
        P_ = np.maximum(P, clip) # Clip probability: if P is very small, then set it to 0.001
        P_ = np.minimum(P_, 1-clip) # Clip probability: if P_ is very big, then set it to 0.999
        P_over_one_minus_P = P_/(1.0-P_)
        #leaf_log_posterior = -np.log(1.0 + P_over_one_minus_P[is_leaf]*phi(X,mu)/phi(X) )
        leaf_log_posterior = -np.log1p( P_over_one_minus_P[is_leaf]*phi(X,mu)/phi(X) )
        

        # calculate posterior for all structures
        # now for each structure, I need a leaf likelihod, and an adjustment
        #posterior = np.zeros((N,M))
        log_posterior = np.zeros((N,M))
        for i in range(M):
            #posterior[:,i] = np.prod(leaf_posterior[:,Descendants_and_self[i,:][is_leaf]],1)
            log_posterior[:,i] = np.sum(leaf_log_posterior[:,Descendants_and_self[i,:][is_leaf]],1)
        
        # calculate adjustment factor for correlations
        Q = Q_from_P(P,A)
        #adjustment_single = np.ones(M)
        log_adjustment_single = np.zeros(M)
        for i in range(M):
            if is_leaf[i]:
                continue
            #adjustment_single[i] = (1.0 - P[i])/ ((1.0 - P[i]) + P[i]*np.prod(1.0 - Q[A[i,:]]))
            #log_adjustment_single[i] = -np.log(1.0 + P_over_one_minus_P[i]*np.prod(1.0 - Q[A[i,:]]))
            log_adjustment_single[i] = -np.log1p(P_over_one_minus_P[i]*np.prod(1.0 - Q[A[i,:]]))
            
        
        # now my adjust ment requres products of all descendants
        #adjustment = np.ones(M)
        log_adjustment = np.ones(M)
        for i in range(M):
            #adjustment[i] = np.prod(adjustment_single[Descendants_and_self[i,:]])
            log_adjustment[i] = np.sum(log_adjustment_single[Descendants_and_self[i,:]])
            

        # calculate the adjusted posterior
        #posterior = posterior*adjustment
        log_posterior = log_posterior + log_adjustment
        
        #P = np.sum(1.0 - posterior,0)/N        
        #P = np.sum(1.0 - np.exp(log_posterior),0)/N
        P = -np.sum(np.expm1(log_posterior),0)/N
        posterior = np.exp(log_posterior)
        
        # draw        
        if draw>0 and ( (not it%draw) or (it==niter-1)):     
            
            ax[0,0].cla()
            ax[0,0].imshow(posterior, vmin = 0, vmax = 1)
            ax[0,0].set_aspect('auto')
            ax[0,0].set_title('P[Z=0|X] (prob not affected)')
            ax[0,0].set_xticks(np.arange(M))
            ax[0,0].set_xticklabels(names,rotation=15, fontsize = 5)
            ax[0,0].set_ylabel('Sample')

            ax[0,1].cla()
            ax[0,1].bar(np.arange(M),P)
            ax[0,1].set_xticks(np.arange(M))
            ax[0,1].set_xticklabels(names,rotation=15, fontsize = 5)
            ax[0,1].set_ylim((0, 1))

            f.canvas.draw()
    return P


# In[ ]:


def generate_simulated_data(filename, subset, case, n_repeats, N, mu):
    ''' Function that generates simulated data
    
    This function generates the simulated data to be used for permutation testing
    
    Parameters
    ----------
    filename: string
        The user-specified filename to save the generated data; make sure to give it a name that's different 
        from the file name you'll specify for permutation testing results so they're separate files and won't 
        be overwritten.
    
    subset: pandas.DataFrame
        The subset adjacency matrix
    
    case: int
        The case number (1 = nothing is affected, 2 = left hippocampus is affected, 3 = both left hippocampus 
        and left amygdala are affected, 4 = either left hippocampus or left amygdala is affected but not both)
    
    n_repeats: int
        The number of repeats. We want to generate a random dataset with the same parameters but `n_repeats` 
        different realizations of the random variables.
    
    N: int
        The number of samples
    
    mu: float
        The difference in means (assume it's known)
    
    Returns
    ----------
    npz file (written to disk, not explicitly returned)
        The 1st array is X (probability of being affected for each sample), the 2nd array is Z (which 
        structures are affected or unaffected for each sample), and the 3rd array is G (whether each sample is 
        actually affected)
  
    '''
    
    M = subset.shape[0] # Number of total unique structures
    
    for j in range(n_repeats):
        outputs = [] # Empty list for each iteration
        Z = np.zeros((N,M)) # Initialize Z, which will be a binary variable that tells us if a structure is affected
        Naffected = N // 2 # Don't set Naffected to 0 or else there won't be any samples
        number_of_leaves = np.count_nonzero(np.sum(subset, 1) == 0) # Number of leaf structures (zero children)
        
        if case == 1:
            pass
        elif case == 2:
            for i in range(N):
                if i < Naffected: # Assume that the first half of samples are affected and second half are not
                    Z[i][6] = 1 # Left hippocampus is affected
        elif case == 3:
            for i in range(N):
                if i < Naffected: # Assume that the first half of samples are affected and second half are not
                    Z[i][6] = 1 # Left hippocampus is affected
                    Z[i][7] = 1 # Left amygdala is affected
        elif case == 4:
            for i in range(N):
                if i < Naffected: # Assume that the first half of samples are affected and second half are not
                    if np.random.rand() < 0.5:
                        Z[i][6] = 1 # Left hippocampus is affected
                    else:
                        Z[i][7] = 1 # Left amygdala is affected
        
            
        is_leaf = np.concatenate([np.ones(number_of_leaves), np.zeros(M - number_of_leaves)]) # 1 for leaf structures, 0 for non-leaf structures
        is_leaf = np.array(is_leaf, dtype = bool) # Convert is_leaf to the boolean type
        is_leaf = is_leaf[::-1] # Data specific
        m = np.sum(is_leaf) # Number of leaf structures (m = 2)
            
        G = np.arange(N) < Naffected # All falses since all samples are unaffected
        X = Z[:, is_leaf > 0] * mu + np.random.randn(N, m)
        
        np.savez(filename, X = X, Z = Z, G = G)


# In[ ]:


def sorting_function(input_string, dictionary):
    ''' Function that sorts results by posterior probability
    
    This function sorts a list of strings by posterior probability, where parents come before children when 
    there are ties.
    
    Parameters
    ----------
    input_string: list
        A list of strings to be sorted; each string contains structure name, posterior probability, p-value
    
    dictionary: dictionary
        A dictionary whose keys are the structures in the subset and values are 1 through the number of 
        structures in the subset (must be in order from parents to children)
    
    Returns
    ----------
    list
        A sorted list of strings (sorted by posterior probability, where parents come before children when 
        there are ties)
    
    '''
    
    my_list = input_string
    my_list = sorted(my_list, key = lambda x : dictionary[x.split(",")[0]])  
    my_list = sorted(my_list, key = lambda x : float(x.split(",")[1].split("=")[-1]), reverse = True) 
    return my_list


# In[ ]:


def permutation_testing(filename_old, filename_new, subset, n_repeats, nperm, N, mu, niter, clip, initial_prob):
    ''' Function that conducts permutation testing
    
    This function conducts permutation testing using the generated data
    
    Parameters
    ----------
    filename_old: string
        The filename of the generated data
    
    filename_new: string
        The user-specified filename for the permutation testing results (choose a different name from 
        filename_old if you don't want generated data to get overwritten by permutation testing results)
    
    subset: pandas.DataFrame
        The subset adjacency matrix
    
    n_repeats: int
        The number of repeats. We generated a random dataset with the same parameters but `n_repeats` 
        different realizations of the random variables. `n_repeats` must be the same value as `n_repeats`
        when we generated data earlier.
    
    nperm: int
        The number of permutations for permutation testing 
    
    N: int
        The number of samples. N must be the same value as N from generating data earlier.
    
    mu: float
        The difference in means (generally unknown). mu should be the same value as mu from generating data
        earlier in order to get meaningful results. But, mu doesn't have to be the same if you don't want to
        make it the same.
    
    niter: int
        The number of iterations of the EM algorithm
    
    clip: float
        Number that clips probabilities away from 0 or 1
    
    initial_prob: float
        The intial probability
    
    Returns 
    ----------
    npz file (written to disk, not explicitly returned)
        The 1st array contains p-values, 2nd array contains the names of the structures in the subset, 3rd 
        array contains the posterior probabilities, and 4th array contains the information from the prior 3 
        arrays saved in 1 string per structure.
    
    '''
    
    M = subset.shape[0] # Number of total unique structures
    S = np.array(subset, dtype = bool)
    names_subset = subset.columns # List of the 8 structures' names
    Descendants = adjacency_descendants(subset, N=N, mu=mu)
    Descendants_and_self = np.logical_or(Descendants, np.eye(M))
    
    # Load the generated data
    data = np.load(filename_old)
    X = data["X"]
    Z = data["Z"]
    G = data["G"]
    
    for j in range(n_repeats):
        outputs = [] # Empty list for each iteration
        
        ### PARAMETER ESTIMATION ###
    
        P_subset = np.ones(M) * 0.5 # Array of 8 copies of 0.5
        Q = Q_from_P(P_subset, S)

        P0 = np.ones(M) * initial_prob
        P_subset = estimate_P(X[G], mu, S, Descendants_and_self, draw=0, P0=P0, niter=niter, names=names_subset, clip=clip)
        # Set draw = 0 to prevent drawing the graphs
        
        ### GENERATING PERMUTED DATA ###
    
        Ps = []
        for n in range(nperm):
            Xp = X[np.random.permutation(N)[G]]
            P_ = estimate_P(Xp,mu,S,Descendants_and_self,draw=0,niter=niter,P0=P0)
            Ps.append(P_)

        Ps_sort = np.array([np.sort(Pi)[::-1] for Pi in Ps])
        
        ### PERMUTATION TESTING ###
    
        inds = np.argsort(P_subset)[::-1]
        pval = np.zeros_like(P_subset)
        alpha = 0.05
        
        pval_list = [] # Empty list to be filled
        names_list = [] # Empty list to be filled
        posterior_list = [] # Empty list to be filled
        
        for i in range(M):    
            pval[inds[i]] = np.mean(Ps_sort[:,i] >= P_subset[inds[i]])
            outputs.append(f"{names_subset[inds[i]]}, P[Z=1|X]={P_subset[inds[i]]}, p={pval[inds[i]]}")
            # Every structure that gets rejected gets an entry
            
            pval_list.append(pval[inds[i]])
            names_list.append(names_subset[inds[i]])
            posterior_list.append(P_subset[inds[i]])
        
        ### SORT THE POSTERIOR VALUES ###
        # Use the subset adjacency matrix to create a dictionary
        columns = np.array(subset.columns)
        dictionary = dict(enumerate(columns.flatten(), 1))
        dictionary = dict((value, key) for key, value in dictionary.items()) # Swap the keys and values
        outputs = sorting_function(outputs, dictionary)
        
        ### SAVE DATA ### 
        
        np.savez(filename_new, pval = pval_list, names = names_list, posterior = posterior_list, strings = outputs)


# In[ ]:


def false_positive_rate(subset, file_names, n_repeats):
    ''' False positive rate function
    
    This function calculates the false positive rate after permutation testing
    
    Parameters
    ----------
    subset: pandas.DataFrame
        The subset adjacency matrix
    
    file_names: list
        A list containing the file names of the permutation testing results to be used in the calculation
    
    n_repeats: int
        The number of repeats. We generated a random dataset with the same parameters but `n_repeats` 
        different realizations of the random variables. `n_repeats` must be the same value for each file in the 
        file_name list.
    
    Note: we can only calculate false positive rates for cases 1 and 2; make sure the input files in the 
    file_name list were all created for the same case (either case 1 or case 2).
    
    Returns
    ----------
    float
        The false positive rate from permutation testing
        
    '''
    
    # Use the subset adjacency matrix to create a dictionary
    columns = np.array(subset.columns)
    dictionary = dict(enumerate(columns.flatten(), 1))
    dictionary = dict((value, key) for key, value in dictionary.items()) # Swap the keys and values
    
    # False positive rate calculation
    count = 0
    for i in range(0, len(file_names)):
        reject_p = np.zeros(len(columns)) # For each repeat, assume no structures are affected (null hypothesis)
        file = np.load(file_names[i])
        file = file["strings"] # Values corresponding to the key
        file = sorting_function(file, dictionary)
        
        for j in range(0, len(columns)): # For each structure in each repeat...
            p = file[j].find("p=") # Index of "p" in the string
            pval = float(file[j][p::][2::]) # Extract the p-value
            if pval < 0.05: # If the p-value is < 0.05...
                reject_p[j] = 1
            else: # The test statistic is the probabilities, which were sorted, so stop after the first structure we fail to reject
                break
                
        if any(reject_p > 0): # If there's at least one structure with p < 0.05...
            count += 1 # Add 1 to the false positive count
                
    return (count / n_repeats)


# In[ ]:


def false_negative_rate(subset, file_names, case, n_repeats):
    ''' False negative rate function
    
    This function calculates the false negative rate after permutation testing
    
    Parameters
    ----------
    subset: pandas.DataFrame
        The subset adjacency matrix
    
    file_names: list
        A list containing the file names of the permutation testing results to be used in the calculation
    
    case: int
        The case that was used to generate the data; all files in the file_names list must correspond to the 
        same case.
    
    n_repeats: int
        The number of repeats. We generated a random dataset with the same parameters but `n_repeats` 
        different realizations of the random variables. `n_repeats` must be the same value for each file in the 
        file_name list.
        
    Note: we can only calculate false negative rates for cases 2, 3, and 4; make sure the input files in the 
    file_name list were all created for the same case (either case 2 or case 3 or case 4).
    
    Returns
    ----------
    pandas.DataFrame
        Data frame whose columns are "Structure" and "False negative rate" with one row per structure

    '''
    
    # Use the subset adjacency matrix to create a dictionary
    columns = np.array(subset.columns)
    dictionary = dict(enumerate(columns.flatten(), 1))
    dictionary = dict((value, key) for key, value in dictionary.items()) # Swap the keys and values
    
    # Create a counter for each structure
    counters = [] # Empty list to be filled
    for k in range(0, len(columns)): # For each structure in each repeat...
        counters.append(0) # List of counters, one entry for each structure, each counter starts at 0
    
    # False negative rate calculation
    for i in range(0, len(file_names)):
        reject_p = np.zeros(len(columns)) # For each repeat, assume no structures are affected (null hypothesis)
        file = np.load(file_names[i])
        file = file["strings"] # Values corresponding to the key
        file = sorting_function(file, dictionary)
    
        for j in range(0, len(columns)): # For each structure in each repeat...
            p = file[j].find("p=") # Index of "p" in the string
            pval = float(file[j][p::][2::]) # Extract the p-value
            if pval < 0.05: # If the p-value is < 0.05...
                reject_p[j] = 1
            else: # The test statistic is the probabilities, which were sorted, so stop after the first structure we fail to reject
                break
        
        # Calculate false negative rate the same for cases 3 or 4
        if ((case == 3) or (case == 4)): # Go through every structure since both hippocampus and amygdala are affected
            for k in range(0, len(columns)): # For each structure in each repeat...
                for r, s in zip(reject_p, file):
                    if columns[k] in s and r: # If a given structure name is in the string and we decided to reject H0...
                        counters[k] += 1
        
        # Calculate false negative rate differently for case 2
        elif case == 2:
            for k in range(0, len(columns)): # For each structure in each repeat...
                for r, s in zip(reject_p, file):
                    if "Amyg" in s and r:
                        continue # Don't consider the amygdala structures for case 2
                    if columns[k] in s and r: # If a given structure name is in the string and we decided to reject H0...
                        counters[k] += 1
        
    output = pd.DataFrame({"Structure": columns, "False negative rate": 1 - np.array(counters) / n_repeats})
    
    if case == 2:
        output = output[~output["Structure"].str.startswith("Amyg")] # Omit the rows corresponding to amygdala structures
    
    return output                 

