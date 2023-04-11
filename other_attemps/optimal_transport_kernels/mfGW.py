from ot.optim import cg
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.gromov import init_matrix,gwloss,gwggrad

def mixed_fused_gromov_wasserstein2(M, Cs1, Cs2, p, q, loss_fun='square_loss', alpha=0.5, **kwargs):
    # Adapted from fused_gromov_wasserstein
    # The following options have been removed (for simplicity): G0,log,armijo (set to True)
  
    p, q = list_to_array(p, q)

    p0, q0, Cs10, Cs20, M0 = p, q, Cs1, Cs2, M
    # nx = get_backend(p0, q0, C10, C20, M0)
    nx = get_backend(p0, q0, Cs10[0], Cs20[0], M0)  

    p = nx.to_numpy(p)
    q = nx.to_numpy(q)
    Cs1 = [nx.to_numpy(C10) for C10 in Cs10] # C1 = nx.to_numpy(C10)
    Cs2 = [nx.to_numpy(C20) for C20 in Cs20] # C2 = nx.to_numpy(C20)
    M = nx.to_numpy(M0)
    G0 = p[:, None] * q[None, :]
    
    # constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
    Cs_info=[]
    for C1,C2 in zip(Cs1, Cs2):
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)   
        Cs_info.append((constC, hC1, hC2))

    G0 = p[:, None] * q[None, :]

    def f(G):
        return sum([gwloss(constC, hC1, hC2, G) for constC, hC1, hC2 in Cs_info])  # gwloss(constC, hC1, hC2, G)
 
    def df(G):
        return sum([gwggrad(constC, hC1, hC2, G) for constC, hC1, hC2 in Cs_info]) #gwggrad(constC, hC1, hC2, G)


    T, log_fgw = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=True, log=True, **kwargs)

    fgw_dist = nx.from_numpy(log_fgw['loss'][-1], type_as=Cs10[0])

    T0 = nx.from_numpy(T, type_as=Cs10[0])

    log_fgw['fgw_dist'] = fgw_dist
    log_fgw['u'] = nx.from_numpy(log_fgw['u'], type_as=Cs10[0])
    log_fgw['v'] = nx.from_numpy(log_fgw['v'], type_as=Cs10[0])
    log_fgw['T'] = T0

    if loss_fun == 'square_loss':
        gC1 = 2 * C1 * (p[:, None] * p[None, :]) - 2 * T.dot(C2).dot(T.T)
        gC2 = 2 * C2 * (q[:, None] * q[None, :]) - 2 * T.T.dot(C1).dot(T)
        gC1 = nx.from_numpy(gC1, type_as=Cs10[0])
        gC2 = nx.from_numpy(gC2, type_as=Cs10[0])
        fgw_dist = nx.set_gradients(fgw_dist, (p0, q0, Cs10[0], Cs20[0], M0),
                                    (log_fgw['u'] - nx.mean(log_fgw['u']),
                                    log_fgw['v'] - nx.mean(log_fgw['v']),
                                    alpha * gC1, alpha * gC2, (1 - alpha) * T0))


    return fgw_dist

