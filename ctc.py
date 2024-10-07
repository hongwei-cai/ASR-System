import numpy as np

def compute_forced_alignment(S, input_lengths, Y, output_lengths):
    """Compute the best path and its cost for each sequence in the batch.
    Args:
        S: A 3-D numpy array of shape (B, T, C), where B is a batch of utterances, 
            that is, the slice S[b,:,:] contains the negative log-probabilities for utterance b. 
            T is the maximum number of acoustic features, and C = 1+|V| is the 
            output dimension of the neural network, each integer in [0, |V|a indexes 
            a token in the vocabulary, and 0 is the blank token.
        input_lengths: A 1-D numpy array of shape (B,) containing the number of utterances
            for each sequence before subsampling.
        Y: A 2-D numpy array of shape (B, L), where L is the maximum output length.
        output_lengths: A 1-D numpy array of shape (B,) containing the number of output
            labels for each sequence.
    Returns:
        A 1-D numpy array of shape (B,) containing the cost of optimal alignments for each utterance.
        A 2-D numpy array of shape (B, T) containing the optimal alignments for each utterance.

    Notes:
    Y: The output sequence consists of a series of tokens (like characters or words). For example,
        Y = [“h”, “e”, “l”, “l”, “o”].
    A: An alignment, which is a sequence that matches the length of the input acoustic feature sequence X,
        mapping each frame of X to either a token from Y or a special blank token \phi.
	B(·): A collapse operation that removes consecutive duplicate tokens and blanks \phi from an alignment
        sequence A to produce the final output sequence Y.
    B^{-1}(Y) is the set of all possible alignments A that can collapse (via the function B(·)) into
        the target output sequence Y.
    X = [x1, x2, ..., xT] represents the sequence of acoustic feature vectors, where: T is the total
        number of frames for the input sequence.
    P(A|X) is the probability that the ASR system aligns the input sequence X with a particular sequence A,
        where A is a combination of the actual tokens and blank tokens emitted at each frame.
    P(Y|X) is the probability that the ASR system transcribes the input sequence X as a particular output sequence Y.
    """

    B, T, C = S.shape
    
    best_costs = np.zeros(B, dtype=np.float32)
    paths = np.zeros((B, T), dtype=np.int32)

    for b in range(B):
        T = input_lengths[b]
        L = output_lengths[b]

        logit = S[b, :T, :]
        label = Y[b, :L]

        Y_prime = modified_label_sequence(label)
        Y_prime_length = 2 * L + 1

        dp = initialize_dp_table(T, Y_prime_length, logit, Y_prime[1])

        dp = fill_dp_table(T, Y_prime_length, dp, logit, Y_prime)

        path, best_cost = backtrack_best_path(T, Y_prime_length, dp, Y_prime, logit)

        best_costs[b] = best_cost
        paths[b, :T] = path

    return best_costs, paths


def modified_label_sequence(label):
    """Return a modified label sequence Y' by inserting blanks around each token
        in the original label sequence Y.
    Args:
        label: A 1-D numpy array of shape (L,) containing the label sequence.
    Returns:
        A 1-D numpy array containing the modified label sequence.
    """
    modified_label = [0]
    for token in label:
        modified_label.append(token)
        modified_label.append(0)
    return np.array(modified_label)


def initialize_dp_table(T, modified_length, logits, first_token):

    dp = np.full((T, modified_length), np.inf)
    dp[0, 0] = logits[0, 0]  # First position must be blank
    if modified_length > 1:
        dp[0, 1] = logits[0, first_token]  # Transition to the first real token
    return dp


def fill_dp_table(T, modified_length, dp, logits, modified_label):

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

    path = np.zeros(T, dtype=np.int32)

    best_cost = min(dp[T - 1, modified_length - 1], dp[T - 1, modified_length - 2])

    # Backtrack to find the best path
    if dp[T - 1, modified_length - 1] < dp[T - 1, modified_length - 2]:
        l = modified_length - 1 
    else:
        l = modified_length - 2

    for t in range(T - 1, -1, -1):
        path[t] = modified_label[l]

        if l > 0 and dp[t, l] == dp[t - 1, l - 1] + logits[t, modified_label[l]]:
            l -= 1
        elif l > 1 and dp[t, l] == dp[t - 1, l - 2] + logits[t, modified_label[l]]:
            l -= 2

    return path, best_cost


if __name__ == "__main__":
    
    logits = np.array([
        [[-1.24300644, -1.53148002, -1.82497478, -1.84361387],
        [-2.03160175, -1.70886812, -1.64796946, -1.35971863],
        [-1.62002954, -1.40484963, -1.40301463, -1.76084657],
        [-1.98033258, -1.63302012, -2.09690261, -1.71650425],
        [-1.40877317, -1.82033641, -1.28140467, -1.45487309]],

        [[-1.96385535, -1.50716001, -1.9819731 , -1.83315796],
        [-1.87125787, -1.89124944, -1.56504305, -1.51213544],
        [-1.19393974, -1.39374673, -1.181992  , -1.53134799],
        [-1.60003132, -1.45696074, -1.60324448, -1.86780564],
        [-1.60525462, -1.9207874 , -1.92954589, -1.39051422]],

        [[-1.71517678, -1.52948705, -1.84688996, -1.82338406],
        [-1.47198252, -1.64516467, -1.28080548, -1.78481608],
        [-1.41079472, -1.35956706, -1.5350524 , -1.87975484],
        [-1.70789392, -1.80905966, -1.83021258, -1.41456818],
        [-1.7993792 , -1.77317859, -1.66793738, -1.29085457]],

        [[-1.55366847, -1.26996081, -1.89464267, -1.46784448],
        [-1.64916202, -2.01003219, -1.81682787, -1.43484833],
        [-1.67093085, -1.27157232, -1.09819259, -1.70241746],
        [-1.30096143, -1.81958572, -2.02875027, -1.53187437],
        [-1.9952732 , -1.94708287, -1.50456277, -2.0169981 ]]
    ], dtype=np.float32)

    input_lens = np.array([4, 3, 3, 5])
    Tout = 3
    labels = np.array([
        [1, 0, 0],
        [2, 2, 0],
        [1, 2, 3],
        [1, 2, 3]])
    output_lens = np.array([1, 2, 3, 3])
    best_costs, paths = compute_forced_alignment(-logits, input_lens, labels, output_lens)



    print(best_costs)
    print(paths)
