from model import Model, PiLearnModelU, PiLearnModelDU, PiLearnModelUdelay, \
                  PiLearnModelUdelayMu, PiLearnModelO

_name = 'model_comparison'

def _get_base_model():
    return {'basic model': (Model, {})}

def get_base_model_comparison():
    models = {r'$\Delta \vec{u} = \dot{\vec{u}} - \dot{\vec{u}}_e$':
                    (PiLearnModelDU, {}),
              r'$\Delta \vec{u} = \vec{u} - \vec{u}_e$':
                    (PiLearnModelU, {}),
              r'delay learning with $\vec{u}_e$ updates':
                    (PiLearnModelUdelay, {}),
              r'delay learning with $\vec{u}_e$ updates, $\vec{u}$ '
              r'clipped by $\mu$':
                    (PiLearnModelUdelayMu, {}),
              # r'smoothed observations learning with $\vec{o}$':
                    # (PiLearnModelO, {}),
              }
    title = r'Comparison of different learning rules for $\Pi_e$'
    return _get_base_model() | models, _name, title

def get_delay_comparison(delays=[5, 20, 50], model=PiLearnModelUdelay,
                         base_eta=1e-2):
    models = _get_base_model()
    for d in delays:
        models[f'delay = {d}'] = (model, {'delay': d, 'eta': base_eta/d})
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

