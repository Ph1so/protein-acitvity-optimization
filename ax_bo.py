from ax.service.managed_loop import optimize
import requests

# STEP 1: Define assay function
def do_assay(pH: float, enzyme_conc: float, incubation_time: float, buffer: float):
    """
        Inputs:
            - assay parameters
        Output:
            - plate reader results
    """
    response = requests.get("http://192.168.0.101/do_assay", params={
        "pH": pH,
        "enzyme_conc": enzyme_conc,
        "incubation_time": incubation_time,
        "buffer": buffer
    })
    finished = False
    while not finished:
        response = requests.get("http://192.168.0.101/get_status")
        if response["status"] == "true":
            finished = True

    response = requests.get("http://192.168.0.102/get_results")
    return response["results"]

# STEP 2: Define the search space
parameters = [
    {"name": "pH", "type": "range", "bounds": [6.0, 8.0]},
    {"name": "enzyme_conc", "type": "range", "bounds": [0.01, 0.1]},  # in mM
    {"name": "incubation_time", "type": "range", "bounds": [30.0, 120.0]},  # in minutes
    {"name": "buffer", "type": "choice", "values": ["PBS", "Tris", "HEPES"]},
]

# STEP 3: Run Bayesian Optimization: Ax defaults to GP and EI
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    evaluation_function=lambda p: do_assay(p["pH"], p["enzyme_conc"], p["incubation_time"], p["buffer"]),
    objective_name="assay_score",
    total_trials=20,  # number of evaluations to run
)

# STEP 4: Print results
print("Best Parameters Found:")
print(best_parameters)
print("Objective value at best parameters:")
print(values)
