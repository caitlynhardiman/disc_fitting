
default_params = ['Inclination', 'Stellar Mass', 'Scale Height', 'R_c', 'R_in', 'Flaring Exponent', 'PA', 'Dust α']

def init_tracking(filename, parameters=default_params):
    results_file = open(filename, "a")
    num_params = len(parameters)
    size = 20*num_params + 40
    dash = '-' * size + '\n'
    first_columns = ['Iteration Number', 'Reduced Chi Squared']
    results_file.write(dash)
    results_file.write(dash)
    results_file.write('{:<20s}{:<25s}'.format(first_columns[0], first_columns[1]))
    for param in parameters:
        results_file.write('{:<20s}'.format(param))
    results_file.write('\n')
    results_file.write(dash)
    results_file.write(dash)
    results_file.close()

def record_iterations(filename, parameters, counter, chi_value):
    results_file = open(filename, "a")
    results_file.write('{:<20s}{:<25s}'.format(str(counter[0]), str(round(chi_value, 5))))
    for param_value in parameters:
        results_file.write('{:<20s}'.format(str(param_value)))
    results_file.write('\n')
    results_file.close()

def best_result(filename, parameters, chi_value):
    results_file = open(filename, "a")
    num_params = len(parameters)
    size = 20*num_params + 40
    dash = '-' * size + '\n'
    results_file.write(dash)
    results_file.write('{:<20s}{:<25s}'.format('BEST FIT', str(round(chi_value, 5))))
    for param_value in parameters:
        results_file.write('{:<20s}'.format(str(param_value)))
    results_file.write('\n')
    results_file.write(dash)
    results_file.close()
