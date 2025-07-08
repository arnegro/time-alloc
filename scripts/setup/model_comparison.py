import numpy as np
from model import Model, PiLearnModelU, PiLearnModelDU, PiLearnModelUdelay, \
                  PiLearnModelUdelayMu, PiLearnModelO, PiLearnModelUdelayProb
from setup.models import get_feedback_model_pars

_name = 'model_comparison'

def _get_base_model():
    return {'basic model': (Model, {})}

def get_base_model_comparison():
    models = {r'Gradient descent': (PiLearnModelUdelay, {}),
              r'Gradient descent, clipped by $\vec{\mu}$':
                    (PiLearnModelUdelayMu, {}),
              r'Gradient descent, $\Delta t \to 0$': (PiLearnModelDU, {}),
              r'Bayesian updates': (PiLearnModelUdelayProb, {'sigma_u':None})
              }
    title = r'Comparison of different learning rules for $\hat{\Pi}$'
    return _get_base_model() | models, _name, title

def get_delay_comparison(delays=[5, 20, 50], model=PiLearnModelUdelay,
                         base_eta=1e-3, dt=1e-1):
    models = _get_base_model()
    for d in delays:
        models[f'delay = {d}'] = (model, {'delay': d, 'eta': base_eta,
                                          'dt': min(dt, d/10)})
    title = r'Comparison of different delays for delay learning of ' \
            r'$\Pi_e$ with $\vec{u}_e$ updates'
    return models, _name + '_delays', title

def get_noise_comparison(sigmas=[None, 1, 5, 10], model=PiLearnModelUdelay):
    models = _get_base_model()
    for s in sigmas:
        if s is not None:
            label = rf'$\sigma_u = {s}$'
        else:
            label = 'no noise'
        models[label] = (model, {'sigma_u': s})
    title = r'Comparison of different noise levels of the MEU for delay ' \
            r'learning of $\Pi_e$ with $\vec{u}_e$ updates'
    return models, _name + '_noise', title

def compare_prob(sigma=3, delay=5, eta=1e-2):
    models = _get_base_model()
    kwargs = {'eta': eta, 'delay': delay, 'sigma_u': None}
    models[r'delay learning with $\vec{u}_e$ updates'] = \
                                            (PiLearnModelUdelay, kwargs)
    models[r'bayesian delay learning with $\vec{u}_e$ updates'] = \
                                            (PiLearnModelUdelayProb, kwargs)
    kwargs = kwargs.copy()
    kwargs['sigma_u'] = sigma
    models[r'delay learning with $\vec{u}_e$ updates, '
          rf'$\sigma_u = {sigma}$'] = \
                                            (PiLearnModelUdelay, kwargs)
    models[r'bayesian delay learning with $\vec{u}_e$ updates, '
          rf'$\sigma_u = {sigma}$'] = \
                                            (PiLearnModelUdelayProb, kwargs)
    title = 'Comparison of gradient descent and bayesian ' \
           f'linear regression learning (eta={eta}, delay={delay})'
    return models, _name + '_prob', title

def get_error_comparison(sigmas=[None, 1, 5], delays=[5, 15, 25, 50]):
    models = {}
    for s in sigmas:
        for d in delays:
            for m, model in [('grad. desc.', PiLearnModelUdelay),
                             ('bayes.', PiLearnModelUdelayProb)]:
                models[(s, d, m)] = (model, {'sigma_u': s, 'delay': d})
    title = 'Comparison of learning'
    return models, 'error_comparison', title

def get_error_comparison_multitask(g12s=[0, 5, 10], sigma_u=None,
                                   delays=range(1, 10, 2)):
    models = {}
    model = PiLearnModelUdelay
    for g12 in g12s:
        P, G, mu, g = get_feedback_model_pars(g12=g12, mu3=20)
        for d in delays:
            models[(sigma_u, d, g12)] = (model, {'delay': d, 'P': P, 'G': G,
                                                 'sigma_u': sigma_u, 'mu': mu})
    title = 'Comparison of learning'
    return models, 'error_comparison_multitask', title

def get_error_comparison_du(sigmas=[None, 1, 5], delays=[5, 15, 25, 50],
                            dts=[1e-1, 1e-3], eta=1e-1, dt=1e-2):
    models = {}
    m0, model0 = (r'grad. desc.', PiLearnModelUdelay)
    m1, model1 = (r'grad. desc. $\eta/\Delta t$', PiLearnModelUdelay)
    m2, model2 = (r'grad. desc. $\dot{u}$', PiLearnModelDU)
    for s in sigmas:
        for d in delays:
            models[(s, d, m1)] = (model1, {'sigma_u': s, 'eta': eta/d})
            models[(s, d, m0)] = (model0, {'sigma_u': s, 'eta': eta})
        # for dt in dts:
            # models[(s, dt, m2)] = (model2, {'sigma_u': s, 'eta': eta,
                                            # 'dt': dt})
        models[(s, 0, m1)] = (model2, {'sigma_u': s, 'eta': eta, 'dt': dt})
    title = 'Comparison of learning'
    return models, 'error_comparison-du', title

def get_action_comparison(sigmas=[None, 1, 5]):
    models = {}
    for s in sigmas:
        for m, model in [('grad. desc.', PiLearnModelUdelay),
                        (r'grad. desc. $\mu$', PiLearnModelUdelayMu),
                         ('basic model', Model),
                         ('bayes', PiLearnModelUdelayProb)]:
            models[(s, m)] = (model, {'sigma_u': s})
    title = 'comparison of statistics of actions during learning'
    return models, 'action_comparison', title

def get_learning_basin_comparison(sigmas=[None, 2], delay=24, eta=1e-3):
    models = {}
    for s in sigmas:
        for m, model in [('grad. desc.', PiLearnModelUdelay),
                         ('bayes', PiLearnModelUdelayProb)]:
            models[(s, m)] = (model, {'sigma_u': s, 'delay': delay,
                                      'eta': eta})
    title = 'comparison of basin of attraction of learning procedures'
    return models, 'learning_basin', title
