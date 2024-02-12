

def gen_bax_batch(model, algorithm, batch_size):
    pass

def gen_bax_batch_local():
    pass

def gen_bax_batch_dijkstra(model, algorithm, batch_size=1):
    '''
    NOTE:
        - Need gp model
        - need x_batch for optimizing acquisition function
        - 
    '''
    # gp_params = {"ls": 0.3, "alpha": 4.3, "sigma": 1e-2, "n_dimx": 2}
    # modelclass = GpfsGp
    acqfn_params = {"acq_str": "exe", "n_path": 30}
    # model = modelclass(gp_params, data, verbose=False)
    acqfn = BaxAcqFunction(acqfn_params, model, algorithm)
    acqopt = AcqOptimizer({"x_batch": edge_locs, "remove_x_dups": True})
    x_next = acqopt.optimize(acqfn)
    return x_next

def gen_discobax_toy(model, algorithm, batch_size):
    '''
    '''
    pass