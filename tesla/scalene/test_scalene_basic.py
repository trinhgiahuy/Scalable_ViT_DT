import time
import numpy as np

def matrix_multiply_test():
    print("Starting test matrix multiplication...")
    A = np.random.rand(5000, 5000)
    B = np.random.rand(5000, 5000)
    result = np.dot(A, B)
    print("Matrix multiplication complete.")

if __name__ == "__main__":
    start_time = time.time()
    matrix_multiply_test()
    end_time = time.time()
    print(f"Computation Time: {end_time - start_time} seconds")
