import numpy as np

def compute_forced_alignment(logits, input_lengths, labels, output_lengths):
    B, T, C = logits.shape  # B=batch size, T=time steps (input length), C=number of classes (including blank token)
    
    # Initialize outputs for best costs and paths
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
        modified_label = [0]  # Start with blank (0 is reserved for blank token)
        for token in label:
            modified_label.append(token)
            modified_label.append(0)  # Insert blank after each token
        modified_label = np.array(modified_label)
        modified_length = 2 * L + 1  # Length of Y'
        
        # Initialize DP table for costs
        dp = np.full((T, modified_length), np.inf)  # Inf cost for unreachable positions
        dp[0, 0] = logits[b, 0, 0]  # First position must start with blank
        if modified_length > 1:
            dp[0, 1] = logits[b, 0, modified_label[1]]  # Transition to the first real token
            
        # Fill DP table
        for t in range(1, T):
            for l in range(modified_length):
                # Costs from previous frame (horizontal and diagonal transitions)
                cost = dp[t - 1, l]  # Horizontal move
                if l > 0:
                    cost = min(cost, dp[t - 1, l - 1])  # Diagonal move from l-1
                if l > 1 and modified_label[l] != modified_label[l - 2]:
                    cost = min(cost, dp[t - 1, l - 2])  # Diagonal move from l-2 (only if no repeated tokens)
                
                # Update cost for the current frame and label
                dp[t, l] = cost + logits[b, t, modified_label[l]]
        
        # Backtrack to find the best alignment path
        path = np.zeros(T, dtype=np.int32)
        best_cost = min(dp[T - 1, modified_length - 1], dp[T - 1, modified_length - 2])
        l = modified_length - 1 if dp[T - 1, modified_length - 1] < dp[T - 1, modified_length - 2] else modified_length - 2
        
        for t in range(T - 1, -1, -1):
            path[t] = modified_label[l]
            if l > 0 and dp[t, l] == dp[t - 1, l - 1] + logits[b, t, modified_label[l]]:
                l -= 1
            elif l > 1 and dp[t, l] == dp[t - 1, l - 2] + logits[b, t, modified_label[l]]:
                l -= 2
        
        # Store the results
        best_costs[b] = best_cost
        paths[b, :T] = path
    
    return best_costs, paths