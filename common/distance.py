import chainer
import chainer.functions as F
from chainer.backends import cuda


def kl_binary(p_logit, q_logit):
    if isinstance(p_logit, chainer.Variable):
        xp = cuda.get_array_module(p_logit.data)
    else:
        xp = cuda.get_array_module(p_logit)
    p_logit = F.concat([p_logit, xp.zeros(p_logit.shape, xp.float32)], 1)
    q_logit = F.concat([q_logit, xp.zeros(q_logit.shape, xp.float32)], 1)
    return kl_categorical(p_logit, q_logit)


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit)
    _kl = F.sum(p * (F.log_softmax(p_logit) - F.log_softmax(q_logit)), 1)
    return F.mean(_kl)


def kl(p_logit, q_logit):
    if p_logit.shape[1] == 1:
        return kl_binary(p_logit, q_logit)
    else:
        return kl_categorical(p_logit, q_logit)


def distance(p_logit, q_logit, dist_type="KL"):
    if dist_type == "KL":
        return kl(p_logit, q_logit)
    else:
        raise NotImplementedError


def get_unit_vector(d, xp):
    d /= xp.sqrt(1e-6 + xp.sum(d ** 2))
    return d


def vat_loss(model, x_data, y_predict, train=True, epsilon=8.0, xi=1e-6, Ip=1):
    xp = model.xp
    d = xp.random.normal(size=x_data.shape)
    d = get_unit_vector(d, xp).astype(xp.float32)
    for ip in range(Ip):
        x_diff = chainer.Variable(x_data + xi * d)
        y_diff = model(x_diff)
        kl_loss = distance(y_predict, y_diff)
        kl_loss.backward()
        d = x_diff.grad * d
        d = get_unit_vector(d, xp)
    x_adv = x_data + epsilon * d
    y_adv = model(x_adv)
    lsd = kl_categorical(y_predict, y_adv)
    return lsd
