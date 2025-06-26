from flask import Flask, request, jsonify
import requests
from  datetime import datetime
import pytz
import threading
import pickle
import math
import matplotlib.pyplot as plt

from ax.service.managed_loop import optimize
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

app = Flask(__name__)

GENERA_IP = ""
READ_PATH = ""
REPORT_PATH = "report.txt"
TRIAL = 1
TESTING = True

est = pytz.timezone("US/Eastern")

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"result:": "test"})

def plot_optimization_results(experiment, filename_prefix="bo_plot"):
    trial_indices = []
    objective_values = []

    for i, trial in enumerate(experiment.trials.values()):
        if trial.arm and trial.objective_mean is not None:
            trial_indices.append(i + 1)  # Trial numbers start at 1
            objective_values.append(trial.objective_mean)
        elif trial.arm and trial.run and trial.run.metrics:
            # Fallback if objective_mean is not set
            objective = trial.run.metrics.get("assay_score")
            if objective and "mean" in objective:
                trial_indices.append(i + 1)
                objective_values.append(objective["mean"])

    plt.figure(figsize=(10, 5))
    plt.plot(trial_indices, objective_values, marker="o", linestyle="-", color="blue")
    plt.title("Optimization Progress")
    plt.xlabel("Trial Number")
    plt.ylabel("Assay Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png")
    plt.show()

def synthetic_assay_score(pH: float, enzyme_conc: float, incubation_time: float) -> float:
    """
        Expected params for global max:
            pH = 7.4
            enzyme_conc = 0.055
            incubation_time = 84.0
    """
    # Normalize inputs
    pH_norm = (pH - 6.0) / (8.0 - 6.0)  # range 0 to 1
    enzyme_norm = (enzyme_conc - 0.01) / (0.1 - 0.01)  # range 0 to 1
    time_norm = (incubation_time - 30.0) / (120.0 - 30.0)  # range 0 to 1

    # Introduce some non-linearity and multiple peaks
    score = (
        math.sin(3 * math.pi * pH_norm) * math.exp(-((pH_norm - 0.7) ** 2) / 0.1) +  # local max near pH ~ 7.4
        math.cos(5 * math.pi * enzyme_norm) * math.exp(-((enzyme_norm - 0.5) ** 2) / 0.2) +  # local max at mid enzyme conc
        math.sin(2 * math.pi * time_norm) * math.exp(-((time_norm - 0.6) ** 2) / 0.2)  # global max near time ~ 84 mins
    )

    # Normalize result to be in range [0, 100]
    normalized_score = (score + 3) / 6 * 100  # because min ≈ -3, max ≈ +3
    return normalized_score

def get_read_file(filepath: str, pH: float, enzyme_conc: float, incubation_time: float) -> dict: 
    global TRIAL
    result = 0
    if TESTING:
        result = synthetic_assay_score(pH, enzyme_conc, incubation_time)

    with open(REPORT_PATH, "a") as f:
        iteration = ""
        
        # Decimal precision
        ph_decimals = 2
        enzyme_decimals = 4
        time_decimals = 1
        result_decimals = 2
        
        # Column widths
        ph_width = max(len("pH"), len(f"{8.0:.{ph_decimals}f}"))
        enzyme_width = max(len("enzyme_conc"), len(f"{0.1:.{enzyme_decimals}f}"))
        time_width = max(len("incubation_time"), len(f"{120.0:.{time_decimals}f}"))
        result_width = max(len("result"), len(f"{100.0:.{result_decimals}f}"))
        trial_width = max(len("Trial#"), len(str(20)))  # Assuming max 20 trials
        
        if TRIAL == 1:
            iteration += (
                f"{'Trial#':<{trial_width}} "
                f"{'pH':<{ph_width}} "
                f"{'enzyme_conc':<{enzyme_width}} "
                f"{'incubation_time':<{time_width}} "
                f"{'result':<{result_width}} "
                f"Timestamp\n"
            )
            iteration += (
                f"{'-'*trial_width} "
                f"{'-'*ph_width} "
                f"{'-'*enzyme_width} "
                f"{'-'*time_width} "
                f"{'-'*result_width} "
                f"---------\n"
            )
        
        timestamp = datetime.now(est).strftime('%A %I:%M%p (%d/%m/%Y)')
        iteration += (
            f"{TRIAL:<{trial_width}} "
            f"{pH:<{ph_width}.{ph_decimals}f} "
            f"{enzyme_conc:<{enzyme_width}.{enzyme_decimals}f} "
            f"{incubation_time:<{time_width}.{time_decimals}f} "
            f"{result:<{result_width}.{result_decimals}f} "
            f"{timestamp}\n"
        )
        
        TRIAL += 1
        f.write(iteration)

    return result

def run_assay(pH: float, enzyme_conc: float, incubation_time: float):    
    if not TESTING:
        response = requests.get(GENERA_IP, params={
            "pH": pH,
            "enzyme_conc": enzyme_conc,
            "incubation_time": incubation_time,
        })

        finished = False
        while not finished:
            response = requests.get(GENERA_IP + "/get_status")
            if response["status"] == "true":
                finished = True
    
    return get_read_file(READ_PATH, pH, enzyme_conc, incubation_time)

def main():
    # Define the search space
    parameters = [
        {"name": "pH", "type": "range", "bounds": [6.0, 8.0]},
        {"name": "enzyme_conc", "type": "range", "bounds": [0.01, 0.1]},
        {"name": "incubation_time", "type": "range", "bounds": [30.0, 120.0]},
    ]

    # Run Bayesian Optimization
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=lambda p: run_assay(p["pH"], p["enzyme_conc"], p["incubation_time"]),
        objective_name="assay_score",
        total_trials=50,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save experiment
    save_experiment(experiment, f"experiment_{timestamp}.json")
    plot_optimization_results(experiment, filename_prefix=f"bo_plot_{timestamp}")

    # Save complete data with pickle
    save_data = {
        'experiment': experiment,
        'model': model,
        'best_parameters': best_parameters,
        'best_values': values,
        'parameters_config': parameters,
        'timestamp': timestamp
    }
    
    with open(f"complete_optimization_{timestamp}.pkl", "wb") as f:
        pickle.dump(save_data, f)

    # Print and save results
    print("Best Parameters Found:")
    print(best_parameters)
    print("Objective value at best parameters:")
    print(values)
    
    # Also append to your report file
    with open(REPORT_PATH, "a") as f:
        f.write(f"\n=== OPTIMIZATION COMPLETE ===\n")
        f.write(f"Best Parameters: {best_parameters}\n")
        f.write(f"Best Value: {values}\n")
        f.write(f"Files saved: experiment_{timestamp}.json, complete_optimization_{timestamp}.pkl\n")

if __name__ == "__main__":
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=lambda: app.run(host="192.168.0.102", port=8000))
    flask_thread.daemon = True  # Dies when main program exits
    flask_thread.start()
    
    # Now run your optimization
    main()
