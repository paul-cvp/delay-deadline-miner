import numpy as np

class Functions():

    def exp_pdf(self, x, tau):
        """Exponential with lifetime tau"""
        return 1.0 / tau * np.exp(-x / tau)

    def gauss_pdf(self, x, mu, sigma):
        """Gaussian"""
        return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

    def N_gauss_pdf(self, x, N, mu, sigma,binwidth=100):
        """Gaussian"""
        return N * binwidth * self.gauss_pdf(x, mu, sigma)

    def log_gauss_pdf(self, x, mu, sigma):
        return 1.0 / np.sqrt(2 * np.pi) / (sigma * x) * np.exp(-0.5 * (np.log(x) - mu) ** 2 / sigma ** 2)

    def N_log_gauss_pdf(self, x, N, mu, sigma,binwidth=100):
        return N * binwidth * self.log_gauss_pdf(x, mu, sigma)

    def double_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2,binwidth=100):
        return N * binwidth * self.gauss_pdf(x, mu, sigma) + N2 * binwidth * self.gauss_pdf(x, mu2, sigma2)

    def triple_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3,binwidth=100):
        return N * binwidth * self.gauss_pdf(x, mu, sigma) + N2 * binwidth * self.gauss_pdf(x, mu2,
                                                                                  sigma2) + N3 * binwidth * self.gauss_pdf(x,
                                                                                                                      mu3,
                                                                                                                      sigma3)

    def triple_gauss_gauss_log(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3,binwidth=100):
        return N * binwidth * self.gauss_pdf(x, mu, sigma) + N2 * binwidth * self.gauss_pdf(x, mu2,
                                                                                  sigma2) + N3 * binwidth * self.log_gauss_pdf(
            x, mu3, sigma3)

    # def triple_gauss_log_log(x,N,mu,sigma,N2,mu2,sigma2,N3,mu3,sigma3):
    #     return N*binwidth*gauss_pdf(x,mu,sigma) + N2*binwidth*log_gauss_pdf(x,mu2,sigma2) + N3*binwidth*log_gauss_pdf(x,mu3,sigma3)

    def triple_gauss_log_log(self, x, N, mu, sigma, N2, mu2, sigma2, N3, mu3, sigma3,binwidth=100):
        if x < 61:
            return N * binwidth * self.gauss_pdf(x, mu, sigma)
        else:
            return N2 * binwidth * self.log_gauss_pdf(x, mu2, sigma2) + N3 * binwidth * self.log_gauss_pdf(x, mu3, sigma3)

    def double_log_gaussian(self, x, N, mu, sigma, N2, mu2, sigma2,binwidth=100):
        return N * binwidth * self.log_gauss_pdf(x, mu, sigma) + N2 * binwidth * self.log_gauss_pdf(x, mu2, sigma2)

    def double_log_gaussian_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2,binwidth=100):
        return N_exp * binwidth * self.exp_pdf(x, tau) + N * binwidth * self.log_gauss_pdf(x, mu,
                                                                                 sigma) + N2 * binwidth * self.log_gauss_pdf(
            x, mu2, sigma2)

    def double_gaussian_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2,binwidth=100):
        return N_exp * binwidth * self.exp_pdf(x, tau) + N * binwidth * self.gauss_pdf(x, mu, sigma) + N2 * binwidth * self.gauss_pdf(
            x, mu2, sigma2)

    def gaus_log_gauss_exp(self, x, N_exp, tau, N, mu, sigma, N2, mu2, sigma2,binwidth=100):
        return N_exp * binwidth * self.exp_pdf(x, tau) + N * binwidth * self.gauss_pdf(x, mu,
                                                                             sigma) + N2 * binwidth * self.log_gauss_pdf(x,
                                                                                                                    mu2,
                                                                                                                    sigma2)