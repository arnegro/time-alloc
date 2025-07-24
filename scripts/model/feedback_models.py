import numpy as np
from model.model_base import Model

class FeedbackModelG(Model):
    def __init__(self, *args, tau=None, beta=None, g=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau if tau is not None else np.ones_like(self.u)
        self.beta = beta if beta is not None else np.zeros_like(self.u)
        self.g_star = g if g is not None else np.zeros_like(self.u)
        self.g = g if g is not None else np.zeros_like(self.u)
        self._nrecvars = 3

    def simulate(self, *args, **kwargs):
        t, res = self._simulate(*args, g=lambda t : 0, **kwargs)
        a, u, g = res[:,0], res[:,1], res[:,2]
        return t, a, u, g

    def step_g(self, g, a, tau, g_star, beta):
        a = np.clip(a, a_min=0, a_max=None)
        dg = (g_star - g + beta*a) / tau * self.dt
        return g + dg

    def step(self, t, __g):
        super().step(t, self.g)
        self.g = self.step_g(self.g, self.a, self.tau, self.g_star, self.beta)
        return self.a, self.u, self.g

class FeedbackModelPi(Model):
    def __init__(self, *args, tau=None, beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau if tau is not None else np.ones_like(self.u)
        self.beta = beta if beta is not None else np.zeros_like(self.u)
        self.Pi_star = self.Pi.copy()
        self.Pi = self.Pi.copy()
        self._nrecvars = 3

    def simulate(self, *args, **kwargs):
        t, res = self._simulate(*args, **kwargs)
        a, u, Pii = res[:,0], res[:,1], res[:,2]
        return t, a, u, Pii

    def step_Pi(self, Pi, a, tau, Pi_star, beta):
        a = np.clip(a, a_min=0, a_max=None)
        dPii = (np.diag(Pi_star - Pi) + beta*a) / tau * self.dt
        return Pi + np.diag(dPii)

    def step(self, t, __g):
        super().step(t, self.g)
        self.Pi = self.step_Pi(self.Pi, self.a, self.tau, self.Pi_star,
                               self.beta)
        return self.a, self.u, np.diag(self.Pi)

class FeedbackModelMu(Model):
    def __init__(self, *args, tau=None, beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau if tau is not None else np.ones_like(self.u)
        self.beta = beta if beta is not None else np.zeros_like(self.u)
        self.mu_star = self.mu.copy()
        self.mu = self.mu.copy()
        self._nrecvars = 3

    def simulate(self, *args, **kwargs):
        t, res = self._simulate(*args, **kwargs)
        a, u, mu = res[:,0], res[:,1], res[:,2]
        return t, a, u, mu

    def step_Pi(self, mu, a, tau, mu_star, beta):
        a = np.clip(a, a_min=0, a_max=None)
        dmu = (mu_star - mu + beta*a) / tau * self.dt
        return mu + dmu

    def step(self, t, __g):
        super().step(t, self.g)
        self.mu = self.step_mu(self.mu, self.a, self.tau, self.mu_star,
                               self.beta)
        return self.a, self.u, self.mu

class GFeedbackExtension:
    def __init__(self, g_bar, tau, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_bar = g_bar
        self.tau = tau
        self.beta = beta
        self._g = self.g_bar.copy()
        self.g_bias = np.zeros_like(self.u)
        self._nrecvars += 1

    def simulate(self, *args, g0=None, **kwargs):
        self._g = self.g_bar.copy() if g0 is None else g0
        return super().simulate(*args, **kwargs)

    def step(self, t, g):
        a = np.clip(self.a, a_min=0, a_max=None)
        self._g += (self.g_bar - self._g + self.beta * a) / self.tau * self.dt
        self.g_bias = self._g - self.g_bar
        res = super().step(t, self._g + g)
        return *res, self._g

    @staticmethod
    def make(cls, *args, **kwargs):
        cls_ext = type(f'{cls.__name__}-g-feedback-extended',
                       (GFeedbackExtension, cls), {})
        return cls_ext(*args, **kwargs)
