import numpy as np
from matplotlib import pyplot as plt
from model import PiLearnModelUdelay
from setup.models import get_feedback_model_pars, get_base_pars
import types

def my_record_setup(self):
    self.P_est_hist = []
def my_record(self):
    self.P_est_hist.append(self.P_est.copy())

def get_angles_magnitudes(P, P_est_hist, agg):
    dP = np.diff(P_est_hist, axis=0)
    k = int(len(dP) / agg) * agg
    dP = dP[:k]
    dP_chunks = dP.reshape(-1, agg, dP.shape[1], dP.shape[1])
    dP_agg = dP_chunks.sum(axis=1)

    dP_opt = P - np.array(model.P_est_hist)[:k][::agg]
    A_flat = dP_agg.reshape(len(dP_agg), -1)
    B_flat = dP_opt.reshape(len(dP_opt), -1)

    dot_products = np.einsum('ij,ij->i', A_flat, B_flat)

    # Norms
    A_norms = np.linalg.norm(A_flat, axis=1)
    B_norms = np.linalg.norm(B_flat, axis=1)

    # Cosine of angles
    cos_theta = dot_products / (A_norms * B_norms)

    # Clip to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute angles in radians or degrees
    angles = np.arccos(cos_theta)
    angles_deg = np.degrees(angles)

    magnitudes = A_norms / B_norms
    return angles, magnitudes

if __name__ == '__main__':
    P, G, mu, g = get_feedback_model_pars()
    P, G, mu, g = get_base_pars()
    aggT = 50
    T = 25*aggT
    dt = 1e-1
    agg = int(aggT / dt)
    delays = [1, 5, 10, 20]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    for d in delays:
        print(d)
        angles, magnitudes, errors = [], [], []
        for _ in range(10):
            P_est = P + np.random.randn(*P.shape)
            model = PiLearnModelUdelay(P, G, mu, P_est=P_est, delay=d,
                                    sigma_u=None, dt=dt)
            model._record_setup = types.MethodType(my_record_setup, model)
            model._record = types.MethodType(my_record, model)
            t, a, u, u_est = model.simulate(g, T=T)
            # print(np.mean(np.clip(a, a_min=0, a_max=None)), d*aggT)
            _angles, _magnitudes = get_angles_magnitudes(P, model.P_est_hist, agg)
            angles.append(_angles)
            magnitudes.append(_magnitudes)
            errors.append(np.mean((model.P_est_hist - P)**2, axis=(1,2)))
        k = int((len(t)-1) / agg) * agg
        ax1.plot(t[:k][::agg]+aggT, np.nanmedian(angles, axis=0))
        ax1.fill_between(t[:k][::agg]+aggT, 
                         *np.nanquantile(angles, [.1, .9], axis=0), alpha=.2)
        ax2.plot(t[:k][::agg]+aggT, np.nanmedian(magnitudes, axis=0))
        ax2.fill_between(t[:k][::agg]+aggT, 
                         *np.nanquantile(magnitudes, [.1, .9], axis=0), alpha=.2)
        ax3.plot(t, np.median(errors, axis=0), label=d)
        ax3.fill_between(t, *np.nanquantile(errors, [.1, .9], axis=0), alpha=.2)
        """
        print(P_est)
        print(P)
        print(angles)
        print(magnitudes)
        """
    ax3.legend(title='delay [h]')
    plt.show()
