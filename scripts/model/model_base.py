import numpy as np

class Model():
    def __init__(self, P, G, mu, dt=1e-2, a0=None, u0=None, sigma_u=None):
        self.P = P.copy()
        self.G = G.copy()
        self.mu = mu.copy()
        self.dim = len(mu)
        self.a0 = np.zeros(self.dim) if a0 is None else a0.copy()
        self.u0 = np.zeros(self.dim) if u0 is None else u0.copy()
        self.a = self.a0.copy()
        self.u = self.u0.copy()
        self.dt = dt
        self.sigma_u = sigma_u
        self._nrecvars = 2

    def _record(self, t):
        return
    def _record_setup(self):
        return

    def step_a(self, a, u, G, mu):
        da = self.dt * (np.clip(u, a_min=mu, a_max=None)
                      - G @ np.clip(a, a_min=0, a_max=None))
        return a + da

    def step_u(self, a, u, g, P, sigma_u=None):
        du = self.dt * (g - P @ np.clip(a, a_min=0, a_max=None))
        if sigma_u is not None:
            du += np.sqrt(self.dt) * sigma_u * np.random.randn()
        return u + du

    def step(self, t, g):
        self.a = self.step_a(self.a, self.u, self.G, self.mu)
        self.u = self.step_u(self.a, self.u, g, self.P, sigma_u=self.sigma_u)
        self._record(t)
        return self.a, self.u

    def simulate(self, *args, **kwargs):
        return self._simulate(*args, **kwargs)
        # a, u = res[:,0], res[:,1]
        # return t, a, u

    def _simulate(self, g, T, a0=None, u0=None, verbose=True):
        self.a = a0 if a0 is not None else self.a0.copy()
        self.u = u0 if u0 is not None else self.u0.copy()
        self._record_setup()
        t = np.arange(0, T, self.dt)
        res = np.empty((len(t), self._nrecvars, self.dim))
        try:
            g(0)
        except TypeError:
            g = lambda t : g
        for i, _t in enumerate(t):
            if verbose:
                print(f'[{int(np.round(i/len(t)*100))}%] --- '
                      f'{_t:.2f}/{T} hours simulated', end='       \r')
            res[i] = self.step(_t, g(_t))
        if verbose:
                print(f'[100%] --- '
                      f'{T}/{T} hours simulated', end='       \n')
        return t, *res.swapaxes(0, 1)
