import numpy as np

SELECT_RATIO = 0.1

class RandomSelector():
    def __init__(self):
        pass

    def select(self, pmf, train, test, mu, sigma, num, select_ratio=SELECT_RATIO):
        sample_size = int(num * select_ratio) # normal mask

        tosample = np.where(np.isnan(train))
        idx_pairs = list(zip(tosample[0], tosample[1]))

        idx_samples = np.random.choice(len(idx_pairs), sample_size, replace=False)
        return [idx_pairs[i] for i in idx_samples]

class ActiveSelector():
    def __init__(self):
        pass

    def select(self, pmf, train, test, mu, sigma, num, select_ratio=SELECT_RATIO):
        sample_size = int(num * select_ratio) # normal mask

        R = np.zeros((test.shape[0], test.shape[1], 
                      len(pmf.trace.posterior.draw.values)))
        for cnt in pmf.trace.posterior.draw.values:
            U = pmf.trace.posterior["U"].sel(chain=0, draw=cnt)
            V = pmf.trace.posterior["V"].sel(chain=0, draw=cnt)
            sample_R = pmf.predict(U, V) * sigma + mu
            R[:, :, cnt] = sample_R

        # Calculate the uncertainty
        uncertainty = np.std(R, axis=2)
        tosample = np.where(np.isnan(train))
        idx_pairs = list(zip(tosample[0], tosample[1]))
        idx_pairs = sorted(idx_pairs, key=lambda x: -uncertainty[x])

        print(uncertainty[:10])
        print("Uncertainty: ", [uncertainty[i] for i in idx_pairs[:10]])
        print("Uncertainty: ", uncertainty.shape)

        selected = [idx_pairs[idx] for idx in range(sample_size)]

        return selected

class WorstSelector():
    def __init__(self):
        pass

    def select(self, pmf, train, test, mu, sigma, num, select_ratio=SELECT_RATIO):
        sample_size = int(num * select_ratio) # normal mask

        R = np.zeros(test.shape)
        for cnt in pmf.trace.posterior.draw.values:
            U = pmf.trace.posterior["U"].sel(chain=0, draw=cnt)
            V = pmf.trace.posterior["V"].sel(chain=0, draw=cnt)
            sample_R = pmf.predict(U, V)
            R += sample_R
        running_R = R / len(pmf.trace.posterior.draw.values)

        error = np.abs(test - running_R) * sigma
        tosample = np.where(np.isnan(train))
        idx_pairs = list(zip(tosample[0], tosample[1]))
        idx_pairs = sorted(idx_pairs, key=lambda x: -error[x])

        print(error[:10])
        print("Error: ", [error[i] for i in idx_pairs[:10]])
        print("Error: ", error.shape)

        selected = [idx_pairs[idx] for idx in range(sample_size)]

        return selected