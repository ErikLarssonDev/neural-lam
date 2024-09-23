import numpy as np

# Define a function that calculates the difference between P*W*W and X
def find_best_P_W(X, max_range=100):
    best_P = None
    best_W = None
    min_diff = float('inf')  # Initialize with a large value
    
    # Try all integer values of P and W in the range [-max_range, max_range]
    for P in range(-max_range, max_range + 1):
        P=4
        for W in range(-max_range, max_range + 1):
            if W == 0:  # Avoid division by zero issues
                continue
            diff = abs(P * W * W - X)  # Absolute difference from X
            if diff < min_diff:
                min_diff = diff
                best_P = P
                best_W = W
                
    return best_P, best_W, min_diff

# Given value of X
X = 268  # Example value, you can change this
Y = 238

# Call the function to find the best integer P and W
P_opt_x, W_opt_x, min_diff_x = find_best_P_W(X)
P_opt_y, W_opt_y, min_diff_y = find_best_P_W(Y)


print(f"Optimal P: {P_opt_x, P_opt_y}")
print(f"Optimal W: {W_opt_x, W_opt_y}")
print(f"P * W * W = {P_opt_x * W_opt_x * W_opt_x} (Target X: {X})")
print(f"Difference from X: {min_diff_x}")
print(f"P * W * W = {P_opt_y * W_opt_y * W_opt_y} (Target Y: {Y})")
print(f"Difference from Y: {min_diff_y}")

