import numpy as np

class Model():
    def __init__(self, P, G, mu, dt=1e-2, a0=None, u0=None, sigma_u=None):
        self.P = P
        self.G = G
        self.mu = mu
        self.dim = len(mu)
        self.a0 = np.zeros(self.dim) if a0 is None else a0
        self.u0 = np.zeros(self.dim) if u0 is None else u0
        self.a = self.a0.copy()
        self.u = self.u0.copy()
        self.dt = dt
        self.sigma_u = sigma_u
        self._nrecvars = 2

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
        self.u = self.step_u(self.a, self.u, g, self.P)
        return self.a, self.u

    def simulate(self, *args, **kwargs):
        t, res = self._simulate(*args, _nrecvars=2, **kwargs)
        a, u = res[:,0], res[:,1]
        return t, a, u

    def _simulate(self, g, T, a0=None, u0=None, verbose=True, _nrecvars=2):
        self.a = a0 if a0 is not None else self.a0.copy()
        self.u = u0 if u0 is not None else self.u0.copy()
        t = np.arange(0, T, self.dt)
        res = np.empty((len(t), _nrecvars, self.dim))
        for i, _t in enumerate(t):
            if verbose:
                print(f'[{int(np.round(i/len(t)*100))}%] --- '
                      f'{_t:.2f}/{T} hours simulated', end='       \r')
            res[i] = self.step(_t, g(_t))
        if verbose: print()
        return t, res

class PiLearnModelBase(Model):
    def __init__(self, *args, eta=1e-3, P_est=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.P_est = np.zeros((self.dim, self.dim)) \
                                if P_est is None else P_est
        self.eta = eta
        self.u_est = self.u0.copy()

    def step_P_est(self):
        return P_est

    def step(self, t, g):
        self.a = self.step_a(self.a, self.u_est, self.G, self.mu)
        self.u = self.step_u(self.a, self.u, g, self.P, self.sigma_u)
        self.u_est = self.step_u(self.a, self.u_est, g, self.P_est, None)
        self.P_est = self.step_P_est()
        if hasattr(self, 'err'):
            self.err.append(np.mean((self.P_est - self.P)**2))
        return self.a, self.u, self.u_est

    def simulate(self, *args, **kwargs):
        self.err = []
        t, res = self._simulate(*args, _nrecvars=3, **kwargs)
        a, u, u_est = res[:,0], res[:,1], res[:,2]
        return t, a, u, u_est

