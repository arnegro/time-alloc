import numpy as np
from model.model_base import PiLearnModelBase

class PiLearnModelU(PiLearnModelBase):
    def step_P_est(self):
        delta_u = self.u - self.u_est
        a = np.clip(self.a, a_min=0, a_max=None)
        dP = - self.dt * np.outer(delta_u, a) * self.eta
        return self.P_est + dP

class PiLearnModelUdelay(PiLearnModelBase):
    def __init__(self, *args, delay=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = 5
        self.A = np.zeros_like(self.a)
    
    def step_P_est(self):
        delta_u = self.u - self.u_est
        dP = - np.outer(delta_u, self.A) * self.eta
        return self.P_est + dP

    def step(self, t, g):
        self.a = self.step_a(self.a, self.u_est, self.G, self.mu)
        self.A += self.dt * np.clip(self.a, a_min=0, a_max=None)
        self.u = self.step_u(self.a, self.u, g, self.P)
        self.u_est = self.step_u(self.a, self.u_est, g, self.P_est)
        update = t // self.delay != (t+self.dt) // self.delay
        if update:
            self.P_est = self.step_P_est()
            self.u_est = self.u
            self.A[:] = 0
        if hasattr(self, 'err'):
            self.err.append(np.mean((self.P_est - self.P)**2))
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
        return self.a, self.u, self.u_est

class PiLearnModelUdelayMu(PiLearnModelUdelay):
    def step_P_est(self):
        delta_u = np.clip(self.u, a_min=self.mu, a_max=None) \
                - np.clip(self.u_est, a_min=self.mu, a_max=None)
        dP = - np.outer(delta_u, self.A) * self.eta
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
