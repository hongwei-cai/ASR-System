import numpy as np

def initialize_dp_table(T, modified_length, logits, first_token):
    """
    Initialize the dynamic programming table with the first time step.
    """
    dp = np.full((T, modified_length), np.inf)
    dp[0, 0] = logits[0, 0]  # First position must be blank
    if modified_length > 1:
        dp[0, 1] = logits[0, first_token]  # Transition to the first real token
    return dp

def create_modified_label(label):
    """
    Create the modified label sequence by inserting blanks between tokens.
    """
    modified_label = [0]  # Start with a blank token
    for token in label:
        modified_label.append(token)
        modified_label.append(0)  # Insert blank after each token
    return np.array(modified_label)

def fill_dp_table(T, modified_length, dp, logits, modified_label):
    """
    Fill the dynamic programming table by calculating the minimum cost at each time step and label index.
    """
    for t in range(1, T):
        for l in range(modified_length):
            # Horizontal move (stay at the same token)
            cost = dp[t - 1, l]
            # Diagonal move from l-1
            if l > 0:
                cost = min(cost, dp[t - 1, l - 1])
            # Diagonal move from l-2 (only if no repeated tokens)
            if l > 1 and modified_label[l] != modified_label[l - 2]:
                cost = min(cost, dp[t - 1, l - 2])
            # Update cost for the current frame and label
            dp[t, l] = cost + logits[t, modified_label[l]]
    return dp

def backtrack_best_path(T, modified_length, dp, modified_label, logits):
    """
    Backtrack through the dynamic programming table to find the optimal alignment path.
    """
    path = np.zeros(T, dtype=np.int32)
    # Determine the best cost at the final time step
    best_cost = min(dp[T - 1, modified_length - 1], dp[T - 1, modified_length - 2])
    # Backtrack to find the best path
    l = modified_length - 1 if dp[T - 1, modified_length - 1] < dp[T - 1, modified_length - 2] else modified_length - 2
    for t in range(T - 1, -1, -1):
        path[t] = modified_label[l]
        if l > 0 and dp[t, l] == dp[t - 1, l - 1] + logits[t, modified_label[l]]:
            l -= 1
        elif l > 1 and dp[t, l] == dp[t - 1, l - 2] + logits[t, modified_label[l]]:
            l -= 2
    return path, best_cost

def compute_forced_alignment(logits, input_lengths, labels, output_lengths):
    B, T, C = logits.shape  # B=batch size, T=time steps (input length), C=number of classes (including blank token)
    
    best_costs = np.zeros(B, dtype=np.float32)
    paths = np.zeros((B, T), dtype=np.int32)
    
    for b in range(B):
        # Extract the logits and label for the current utterance
        logit = logits[b, :input_lengths[b], :]  # Extract only up to the actual input length
        label = labels[b, :output_lengths[b]]  # Extract the valid portion of the label sequence
        
        # Number of time steps and output length for this utterance
        T = input_lengths[b]
        L = output_lengths[b]
        
        # Create the modified label sequence (Y') with blanks inserted
        modified_label = create_modified_label(label)
        modified_length = 2 * L + 1  # Length of Y'
        
        # Initialize the dynamic programming table
        dp = initialize_dp_table(T, modified_length, logit, modified_label[1])
        
        # Fill the dynamic programming table
        dp = fill_dp_table(T, modified_length, dp, logit, modified_label)
        
        # Backtrack to find the best path
        path, best_cost = backtrack_best_path(T, modified_length, dp, modified_label, logit)
        
        # Store the results
        best_costs[b] = best_cost
        paths[b, :T] = path
    
    return best_costs, paths