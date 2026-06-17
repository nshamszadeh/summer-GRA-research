import numpy as np
import matplotlib.pyplot as plt

"""

input signal: u(n) (N_in sized vector)
output signal: y(n) (N_out sized vector)

input weights: W_in (N_w x N_in matrix)
reservoir: W (N_w x N_w matrix)
output weights: W_out (N_out x N_w matrix)

leaking rate: a (a in (0,1])

update equations:
    x(n)   = (1-a)x(n-1) + ax_o(n)
    x_o(n) = tanh(W_in)

"""

# NOTE: np.dot() generalizes to matrix multiplication, not just vector dot products
class EchoStateNetwork:
    def __init__(self, input_size, reservoir_size, output_size, leaking_rate, spectral_rad=0.9):
        self.reservoir_size = reservoir_size
        self.input_size = input_size
        self.output_size = output_size
        self.leaking_rate = leaking_rate

        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5
        self.W_res *= spectral_rad / np.max(np.abs(np.linalg.eigvals(self.W_res)))

        self.W_in = np.random.rand(reservoir_size, input_size)
        self.W_out = None

    def train(self, input_data, target_data):
        # feed input data through reservoir
        # perform linear regression on output and target data
        reservoir_states = self.feed_reservoir(input_data)
        #pseudo-inverse reservoir state dot product with target data is equivalent to linear least squares approximation
        self.W_out = np.dot(np.linalg.pinv(reservoir_states), target_data)
    
    def predict(self, input_data):
        # feed reservoir with input
        # return output vector (W_out weights are fixed)
        reservoir_states = self.feed_reservoir(input_data)
        return np.dot(reservoir_states, self.W_out)
    
    def feed_reservoir(self, input_data):
        # initialize reservoir states to 0
        reservoir_activations = np.zeros((len(input_data), self.reservoir_size))
        activation_updates    = np.zeros((len(input_data), self.reservoir_size))
        for t in range(1, len(input_data)):
            activation_updates[t] = np.tanh(np.dot(self.W_in, input_data[t]) + np.dot(self.W_res, reservoir_activations[t-1, :]))
            reservoir_activations[t] = (1-self.leaking_rate) * reservoir_activations[t-1] + self.leaking_rate * activation_updates[t]
        return reservoir_activations

if __name__ == "__main__":
    # train RC on a sine wave
    # Note that input_size = 1, since at each timestep sin(t) is a 1d real number (as opposed to nd real vector)
    time = np.arange(0, 20, 0.1)
    noise = 0.1 * np.random.rand(len(time))
    sine_wave_target = np.sin(time)

    # create ESN
    size = 200
    esn = EchoStateNetwork(input_size=1, reservoir_size=size, output_size=size, leaking_rate=0.01, spectral_rad=0.1)
    esn.train(noise[:,None], sine_wave_target[:,None])
    predictions = esn.predict(noise[:,None])

    # plot results
    plt.figure(figsize=(10,6))
    plt.plot(time, sine_wave_target, label='Target Sine Wave', linestyle='--', marker='o')
    plt.plot(time, predictions, label='ESN Prediction', linestyle='--', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Echo State Network Approximation vs. Target Value')
    plt.show()
