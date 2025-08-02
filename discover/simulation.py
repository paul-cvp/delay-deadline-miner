import numpy as np

r = np.random
r.seed(1)

class Simulation:

    def sample_timing_points(self, func_to_fit, func_params, N_samples, max_sample_y, xmin, xmax):
        x_sampled = np.zeros(N_samples)
        y_sampled = np.zeros(N_samples)
        Ntry = 0
        for i in range(N_samples):
            while True:
                Ntry += 1
                sample_y = r.uniform(0, max_sample_y)
                sample_x = r.uniform(xmin, xmax)
                if sample_y < func_to_fit(sample_x, *func_params):
                    break
            x_sampled[i] = sample_x
            y_sampled[i] = sample_y
        return x_sampled, y_sampled