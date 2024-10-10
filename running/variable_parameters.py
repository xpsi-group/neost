def get_variable_params(parametrization, eos, core_start, number_stars):
    if 'PP' == parametrization:
        return get_variable_params_pp(eos, core_start, number_stars)
    elif 'CS' == parametrization:
        return get_variable_params_cs(eos, number_stars)
    else:
        raise ValueError(f'Unknown parametrization {parametrization}')

def get_variable_params_pp(eos, core_start, number_stars):
    variable_params = None
    if 1.1 == core_start:
        variable_params = {'ceft':[eos.min_norm, eos.max_norm],'ceft_in':[eos.min_index, eos.max_index],'gamma1':[1.,4.5],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[1.5,8.3], 'rho_t2':[1.5,8.3]}
    elif 1.5 == core_start:
        variable_params = {'ceft':[eos.min_norm, eos.max_norm],'ceft_in':[eos.min_index, eos.max_index],'gamma1':[0.,8.],'gamma2':[0.,8.],'gamma3':[0.5,8.],'rho_t1':[2.,8.3],'rho_t2': [2.,8.3]}
    else:
        raise ValueError(f'I do not know which parameters to use for core_start = {core_start}')
    variable_params = update_rhoc(variable_params, number_stars)
    return variable_params

def get_variable_params_cs(eos, number_stars):
    variable_params={'ceft':[eos.min_norm, eos.max_norm],'ceft_in':[eos.min_index, eos.max_index],'a1':[0.1,1.5],'a2':[1.5,12.],'a3/a2':[0.05,2.],'a4':[1.5,37.],'a5':[0.1,1.]}
    variable_params = update_rhoc(variable_params, number_stars)
    return variable_params

def update_rhoc(variable_params, number_stars):
    for i in range(number_stars):
        variable_params.update({'rhoc_' + str(i+1):[14.6, 16]})
    return variable_params
