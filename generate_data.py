import numpy as np
import pickle


def generate_density_matrix(d):
    """Generate a random d-dimensional density matrix."""
    # Random complex matrix
    A = np.random.rand(d, d) + 1j * np.random.rand(d, d)
    # Hermitian matrix
    H = A + A.conj().T
    # Positive semi-definite matrix with trace 1
    rho = H @ H.conj().T
    rho /= np.trace(rho)
    return rho


def generate_measurement_operators(d, num_operators):
    """Generate a set of measurement operators for a d-dimensional system."""
    operators = []
    for _ in range(num_operators):
        operators.append(generate_density_matrix(d))
    # Normalizing to make the sum of operators equal to identity
    total = sum(operators)
    normalized_operators = [op / total for op in operators]
    return normalized_operators


def simulate_measurements(rho, operators, N):
    """Simulate the measurement process."""
    probabilities = [np.real(np.trace(op @ rho)) for op in operators]  # Using Born's rule
    # clamp probabilities to [0, 1]
    probabilities = np.clip(probabilities, 0, 1)
    frequencies = [np.random.binomial(N, p) / N for p in probabilities]  # Simulating frequencies
    return probabilities, frequencies


if __name__ == "__main__":
    # Parameters
    d = 2  # Dimension of the quantum system
    N = 1000  # Number of copies of the quantum state for measurements
    num_operators = 10  # Number of measurement operators
    dataset = []  # List of tuples (frequencies, operators, state, probabilities)
    n_data = 10000  # Number of data points
    for _ in range(n_data):
        rho = generate_density_matrix(d)
        operators = generate_measurement_operators(d, num_operators)
        probabilities, frequencies = simulate_measurements(rho, operators, N)
        dataset.append((frequencies, operators, rho, probabilities))

    # Saving the dataset as pkl file
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
