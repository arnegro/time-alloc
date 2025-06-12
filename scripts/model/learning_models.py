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
        # print(self.a, self.u)
        super().step(t, g)
        self.A += self.dt * np.clip(self.a, a_min=0, a_max=None)
        if update:
            self.update_step()
            # quit()
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
    def update_P_est(self):
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

class PiLearnModelUdelayProb(PiLearnModelUdelay):
    def __init__(self, *args, sigma_u=1, U_inv=None, **kwargs):
        super().__init__(*args, sigma_u=sigma_u, **kwargs)
        n = self.u.shape[0]
        self.M = np.zeros((n, n)) 
        self.M = self.P_est.copy()
        self.U_inv = np.eye(n) *1e-2 if U_inv is None else U_inv
        # self.V = np.eye(n

    def simulate(self, *args, u0=None, **kwargs):
        self._previous_u = u0 if u0 is not None else self.u0.copy()
        self._G = np.zeros_like(self.u)
        return super().simulate(*args, u0=u0, **kwargs)

    def step(self, t, g):
        # g *= 0
        self._G += self.dt * g
        return super().step(t, g)

    def update_P_est(self):
        var = self.sigma_u**2 * self.delay
        # print(var)
        delta_u = self.u - self._previous_u
        y = delta_u - self._G
        x = - self.A
        x = x[:,None]
        y = y[:,None]
        # print(x)
        # print(y)
        U_inv = self.U_inv + np.outer(x, x)# / var
        # print(np.round(self.U_inv, decimals=3))
        # print(np.round(U_inv, decimals=3))
        # print(np.round(delta_u, decimals=3))
        # print(np.round(self.A, decimals=3))
        # print(np.round(self._G, decimals=3))
        # print(np.round(self.M, decimals=3))
        # M = np.linalg.inv(U_inv) @ (self.U_inv @ self.M.T
                                  # - np.outer(self.A, delta_u - self._G) / var)
        dM = np.outer(y - self.M @ x, x) @ np.linalg.inv(U_inv)
        # M = M.T
        # print(np.round(dM, decimals=3))
        self.M += dM
        # print(self.M)
        # quit()
        self.U_inv = U_inv
        return self.P.copy()
        return self.M.copy()
