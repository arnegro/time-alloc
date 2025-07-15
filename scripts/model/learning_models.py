import numpy as np
from model.model_base import Model

class PiLearnModelBase(Model):
    def __init__(self, *args, eta=1e-3, P_est=None, g_bias=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.P_est = np.zeros((self.dim, self.dim)) \
                                if P_est is None else P_est
        self.eta = eta
        self.u_est = self.u0.copy()
        self.g_bias = np.zeros_like(self.u) if g_bias is None else g_bias

    def step_P_est(self):
        return P_est

    def step(self, t, g):
        self.a = self.step_a(self.a, self.u_est, self.G, self.mu)
        self.u = self.step_u(self.a, self.u, g, self.P, self.sigma_u)
        self.u_est = self.step_u(self.a, self.u_est, g-self.g_bias,
                                 self.P_est, None)
        self.P_est = self.step_P_est()
        self._record()
        return self.a, self.u, self.u_est

    def _record(self):
        super()._record()
        if hasattr(self, 'err'):
            self.err.append(np.mean((self.P_est - self.P)**2))
    def _record_setup(self):
        super()._record_setup()
        self.err = []

    def simulate(self, *args, uest0=None, **kwargs):
        self.uest0 = uest0 if uest0 is not None else self.u0.copy()
        t, res = self._simulate(*args, _nrecvars=3, **kwargs)
        a, u, u_est = res[:,0], res[:,1], res[:,2]
        return t, a, u, u_est

class PiLearnModelU(PiLearnModelBase):
    def step_P_est(self):
        delta_u = self.u - self.u_est
        a = np.clip(self.a, a_min=0, a_max=None)
        dP = - self.dt * np.outer(delta_u, a) * self.eta
        return self.P_est + dP

class PiLearnModelUdelay(PiLearnModelBase):
    def __init__(self, *args, delay=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self.A = np.zeros_like(self.a)
    
    def step_P_est(self):
        return self.P_est

    def update_P_est(self):
        delta_u = self.u - self.u_est
        dP = - np.outer(delta_u, self.A) * self.eta
        return self.P_est + dP

    def update_step(self):
        self.P_est = self.update_P_est()
        self.u_est = self.u
        self.A[:] = 0

    def step(self, t, g):
        update = t // self.delay != (t+self.dt) // self.delay
        super().step(t, g)
        self.A += self.dt * np.clip(self.a, a_min=0, a_max=None)
        if update:
            self.update_step()
        return self.a, self.u, self.u_est

class PiLearnModelDU(PiLearnModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dudt = None
        self.dudt_est = None

    def step_P_est(self):
        delta_u = self.dudt - self.dudt_est
        a = np.clip(self.a, a_min=0, a_max=None)
        dP = - self.dt * np.outer(delta_u, a) * self.eta
        return self.P_est + dP

    def step(self, t, g):
        self.dudt = g - self.P @ np.clip(self.a, a_min=0, a_max=None)
        self.dudt_est = g - self.P_est @ np.clip(self.a, a_min=0, a_max=None)
        super().step(t, g)
        self.u_est = self.u  # then limit of delay model with eta -> eta/delay?? --- naja...
        return self.a, self.u, self.u_est

class PiLearnModelUdelayMu(PiLearnModelUdelay):
    def update_P_est(self):
        delta_u = np.clip(self.u, a_min=self.mu, a_max=None) \
                - np.clip(self.u_est, a_min=self.mu, a_max=None)
        _Igmu = np.diag(self.u_est > self.mu).astype(float)
        dP = - np.outer(delta_u, _Igmu @ self.A) * self.eta
        return self.P_est + dP

class PiLearnModelO(PiLearnModelBase):
    ## TODO: faulty!!
    def __init__(self, *args, tau=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.o = np.zeros_like(self.u)
        self.o_est = np.zeros_like(self.u)
        self.A = np.zeros_like(self.a)
        self.tau = tau

    def step(self, t, g):
        _o = self.o.copy()
        self.o += (np.clip(self.u, a_min=self.mu, a_max=None) - self.o) / self.tau * self.dt
        self.o_est += (np.clip(self.u_est, a_min=self.mu, a_max=None) - self.o_est) / self.tau * self.dt
        self.A += (self.a - self.A) / self.tau * self.dt
        super().step(t, g)
        return np.array([self.o[0], self.u[0], self.tau*(self.o-_o)[0]/self.dt + self.o[0]]), self.u, self.u_est
        return self.a, self.u, self.u_est

    def step_P_est(self):
        return self.P
        delta_o = self.o - self.o_est
        dP = - np.outer(delta_o, self.A) * self.eta
        return self.P_est + dP

class PiLearnModelUdelayProb(PiLearnModelUdelay):
    def __init__(self, *args, U_inv=None, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.u.shape[0]
        self.M = self.P_est.copy()
        self.U_inv = np.eye(n) *1e-2 if U_inv is None else U_inv

    def simulate(self, *args, u0=None, **kwargs):
        self._previous_u = u0 if u0 is not None else self.u0.copy()
        self._G = np.zeros_like(self.u)
        return super().simulate(*args, u0=u0, **kwargs)

    def step(self, t, g):
        self._G += self.dt * (g - self.g_bias)
        return super().step(t, g)

    def update_P_est(self):
        # var = self.sigma_u**2 * self.delay
        delta_u = self.u - self._previous_u
        y = delta_u - self._G
        x = - self.A
        x = x[:,None]
        y = y[:,None]
        U_inv = self.U_inv + np.outer(x, x)
        dM = np.outer(y - self.M @ x, x) @ np.linalg.inv(U_inv)
        self.M += dM
        self.U_inv = U_inv
        self._previous_u = self.u.copy()
        self._G *= 0
        return self.M.copy()
