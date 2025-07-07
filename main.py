"""
Bayesian Optimization System for Laboratory Assay Parameter Optimization

This module provides an implementation of Bayesian optimization
for optimizing laboratory assay parameters including pH, enzyme concentration,
and incubation time.

Author: Phi Nguyen
Date: 7/7/2025
Version: 1.0.0
"""

import json
import math
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pytz
import requests
from ax.service.managed_loop import optimize
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment
from flask import Flask, jsonify, request


class Config:
    """Configuration settings for the optimization system."""
    
    # Network settings
    GENERA_IP: str = "192.168.0.101"
    GENERA_PORT: int = 8086
    FLASK_HOST: str = "192.168.0.102"
    FLASK_PORT: int = 8000
    
    # Authentication
    GENERA_USERNAME: str = "admin"
    GENERA_PASSWORD: str = "genera"
    
    # File paths
    READ_PATH: str = ""
    REPORT_PATH: str = "report.txt"
    PUBLIC_DIR: str = r"C:\Users\Public"
    SUBDIR_NAME: str = "Public BayesianOpt"
    
    # Optimization settings
    TOTAL_TRIALS: int = 50
    TESTING_MODE: bool = True
    
    # Timezone
    TIMEZONE: str = "US/Eastern"
    
    # Parameter bounds
    PH_BOUNDS: List[float] = [6.0, 8.0]
    ENZYME_CONC_BOUNDS: List[float] = [0.01, 0.1]
    INCUBATION_TIME_BOUNDS: List[float] = [30.0, 120.0]


class GeneraAPIClient:
    """Client for interacting with the Genera API."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session_id: Optional[str] = None
        self.base_url = f"http://{config.GENERA_IP}:{config.GENERA_PORT}"
    
    def authenticate(self) -> None:
        """Authenticate with the Genera API and store session ID."""
        login_data = {
            "username": self.config.GENERA_USERNAME,
            "password": self.config.GENERA_PASSWORD
        }
        
        headers = {
            "accept": "*/*",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/genera/login",
                json=login_data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Extract session ID from Set-Cookie header
            cookie_header = response.headers.get("Set-Cookie", "")
            if "session=" in cookie_header:
                self.session_id = cookie_header.split("session=")[1].split(";")[0]
                print(f"Successfully authenticated. Session ID: {self.session_id}")
            else:
                raise ValueError("Session ID not found in response headers")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to authenticate with Genera API: {e}")
    
    def submit_process(self, process_id: str, name: str) -> str:
        """Submit a process to the Genera scheduler."""
        if not self.session_id:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        process_data = {
            "processId": process_id,
            "name": name,
            "batchCount": 1,
            "priority": 1
        }
        
        headers = {
            "accept": "application/json",
            "cookie": f"session={self.session_id}"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/genera/scheduler/process-tasks",
                json=process_data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            response_data = response.json()
            process_id = response_data.get("id")
            
            if process_id is None:
                raise ValueError("Process ID not found in response")
            
            return process_id
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to submit process: {e}")
    
    def wait_for_process_completion(self, process_id: str, poll_interval: int = 2) -> None:
        """Wait for a process to complete."""
        if not self.session_id:
            raise ValueError("Not authenticated. Call authenticate() first.")
        
        headers = {"cookie": f"session={self.session_id}"}
        
        while True:
            try:
                response = requests.get(
                    f"{self.base_url}/genera/scheduler/process-tasks/{process_id}/state",
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                
                state = response.text.strip('"')
                print(f"Process state: {state}")
                
                if state == "FINISHED":
                    break
                elif state == "ERROR":
                    raise RuntimeError(f"Process {process_id} failed")
                
                time.sleep(poll_interval)
                
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Failed to check process state: {e}")


class AssayOptimizer:
    """Main class for performing Bayesian optimization of assay parameters."""
    
    def __init__(self, config: Config):
        self.config = config
        self.genera_client = GeneraAPIClient(config)
        self.trial_counter = 1
        self.timezone = pytz.timezone(config.TIMEZONE)
        
        # Ensure report directory exists
        Path(config.REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    def synthetic_assay_score(self, pH: float, enzyme_conc: float, incubation_time: float) -> float:
        """
        Synthetic assay score function for testing purposes.
        
        Expected optimal parameters:
            pH = 7.4
            enzyme_conc = 0.055
            incubation_time = 84.0
        
        Args:
            pH: pH value between 6.0 and 8.0
            enzyme_conc: Enzyme concentration between 0.01 and 0.1
            incubation_time: Incubation time between 30.0 and 120.0
            
        Returns:
            Normalized assay score between 0 and 100
        """
        # Normalize inputs to [0, 1] range
        ph_norm = (pH - 6.0) / (8.0 - 6.0)
        enzyme_norm = (enzyme_conc - 0.01) / (0.1 - 0.01)
        time_norm = (incubation_time - 30.0) / (120.0 - 30.0)
        
        # Create complex response surface with multiple peaks
        score = (
            math.sin(3 * math.pi * ph_norm) * math.exp(-((ph_norm - 0.7) ** 2) / 0.1) +
            math.cos(5 * math.pi * enzyme_norm) * math.exp(-((enzyme_norm - 0.5) ** 2) / 0.2) +
            math.sin(2 * math.pi * time_norm) * math.exp(-((time_norm - 0.6) ** 2) / 0.2)
        )
        
        # Normalize to [0, 100] range
        normalized_score = (score + 3) / 6 * 100
        return max(0, min(100, normalized_score))  # Clamp to valid range
    
    def log_trial_result(self, pH: float, enzyme_conc: float, incubation_time: float, result: float) -> None:
        """Log trial results to report file."""
        timestamp = datetime.now(self.timezone).strftime('%A %I:%M%p (%d/%m/%Y)')
        
        # Format specifications
        format_specs = {
            'trial_width': max(len("Trial#"), len(str(self.config.TOTAL_TRIALS))),
            'ph_width': max(len("pH"), len(f"{max(self.config.PH_BOUNDS):.2f}")),
            'enzyme_width': max(len("enzyme_conc"), len(f"{max(self.config.ENZYME_CONC_BOUNDS):.4f}")),
            'time_width': max(len("incubation_time"), len(f"{max(self.config.INCUBATION_TIME_BOUNDS):.1f}")),
            'result_width': max(len("result"), len("100.00"))
        }
        
        with open(self.config.REPORT_PATH, "a", encoding="utf-8") as f:
            # Write header for first trial
            if self.trial_counter == 1:
                header = (
                    f"{'Trial#':<{format_specs['trial_width']}} "
                    f"{'pH':<{format_specs['ph_width']}} "
                    f"{'enzyme_conc':<{format_specs['enzyme_width']}} "
                    f"{'incubation_time':<{format_specs['time_width']}} "
                    f"{'result':<{format_specs['result_width']}} "
                    f"Timestamp\n"
                )
                separator = (
                    f"{'-' * format_specs['trial_width']} "
                    f"{'-' * format_specs['ph_width']} "
                    f"{'-' * format_specs['enzyme_width']} "
                    f"{'-' * format_specs['time_width']} "
                    f"{'-' * format_specs['result_width']} "
                    f"---------\n"
                )
                f.write(header + separator)
            
            # Write trial data
            trial_line = (
                f"{self.trial_counter:<{format_specs['trial_width']}} "
                f"{pH:<{format_specs['ph_width']}.2f} "
                f"{enzyme_conc:<{format_specs['enzyme_width']}.4f} "
                f"{incubation_time:<{format_specs['time_width']}.1f} "
                f"{result:<{format_specs['result_width']}.2f} "
                f"{timestamp}\n"
            )
            f.write(trial_line)
        
        self.trial_counter += 1
    
    def write_parameters_file(self, pH: float, enzyme_conc: float, incubation_time: float) -> None:
        """Write parameters to JSON file for external process consumption."""
        public_dir = Path(self.config.PUBLIC_DIR)
        subdir = public_dir / self.config.SUBDIR_NAME
        subdir.mkdir(parents=True, exist_ok=True)
        
        parameters = {
            "pH": pH,
            "enzyme_conc": enzyme_conc,
            "incubation_time": incubation_time
        }
        
        file_path = subdir / "params.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(parameters, f, indent=4)
    
    def run_assay(self, pH: float, enzyme_conc: float, incubation_time: float) -> float:
        """
        Run a single assay with given parameters.
        
        Args:
            pH: pH value
            enzyme_conc: Enzyme concentration
            incubation_time: Incubation time
            
        Returns:
            Assay score
        """
        if self.config.TESTING_MODE:
            # Use synthetic function for testing
            result = self.synthetic_assay_score(pH, enzyme_conc, incubation_time)
        else:
            # Real assay execution
            self.write_parameters_file(pH, enzyme_conc, incubation_time)
            
            # Submit process to Genera
            process_id = self.genera_client.submit_process(
                process_id="C:\\retisoft\\genera\\processes\\move_plate2.process",
                name="move_plate2"
            )
            
            # Wait for completion
            self.genera_client.wait_for_process_completion(process_id)
            
            # TODO: Read actual result from file
            result = 0.0  # Placeholder
        
        # Log the trial
        self.log_trial_result(pH, enzyme_conc, incubation_time, result)
        
        return result
    
    def plot_optimization_results(self, experiment, filename_prefix: str = "bo_plot") -> None:
        """Plot optimization progress."""
        trial_indices = []
        objective_values = []
        
        for i, trial in enumerate(experiment.trials.values()):
            if trial.arm and trial.objective_mean is not None:
                trial_indices.append(i + 1)
                objective_values.append(trial.objective_mean)
            elif trial.arm and trial.run and trial.run.metrics:
                # Fallback for alternative objective storage
                objective = trial.run.metrics.get("assay_score")
                if objective and "mean" in objective:
                    trial_indices.append(i + 1)
                    objective_values.append(objective["mean"])
        
        if not trial_indices:
            print("Warning: No valid trial data found for plotting")
            return
        
        plt.figure(figsize=(12, 8))
        plt.plot(trial_indices, objective_values, marker="o", linestyle="-", 
                color="blue", linewidth=2, markersize=6)
        plt.title("Bayesian Optimization Progress", fontsize=16, fontweight='bold')
        plt.xlabel("Trial Number", fontsize=12)
        plt.ylabel("Assay Score", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{filename_prefix}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Optimization plot saved as: {plot_path}")
    
    def save_results(self, experiment, model, best_parameters: Dict, best_values: Dict) -> str:
        """Save optimization results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save experiment as JSON
        experiment_file = f"experiment_{timestamp}.json"
        save_experiment(experiment, experiment_file)
        
        # Save complete results as pickle
        save_data = {
            'experiment': experiment,
            'model': model,
            'best_parameters': best_parameters,
            'best_values': best_values,
            'parameters_config': self.get_parameter_config(),
            'timestamp': timestamp,
            'config': self.config
        }
        
        pickle_file = f"complete_optimization_{timestamp}.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(save_data, f)
        
        # Append results to report
        with open(self.config.REPORT_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"OPTIMIZATION COMPLETE - {timestamp}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Best Parameters: {best_parameters}\n")
            f.write(f"Best Value: {best_values}\n")
            f.write(f"Files saved: {experiment_file}, {pickle_file}\n")
            f.write(f"{'='*50}\n")
        
        return timestamp
    
    def get_parameter_config(self) -> List[Dict]:
        """Get parameter configuration for optimization."""
        return [
            {"name": "pH", "type": "range", "bounds": self.config.PH_BOUNDS},
            {"name": "enzyme_conc", "type": "range", "bounds": self.config.ENZYME_CONC_BOUNDS},
            {"name": "incubation_time", "type": "range", "bounds": self.config.INCUBATION_TIME_BOUNDS},
        ]
    
    def run_optimization(self) -> Tuple[Dict, Dict, object, object]:
        """
        Run the complete Bayesian optimization process.
        
        Returns:
            Tuple of (best_parameters, best_values, experiment, model)
        """
        print(f"Starting Bayesian optimization with {self.config.TOTAL_TRIALS} trials...")
        print(f"Testing mode: {self.config.TESTING_MODE}")
        
        # Initialize Genera client if not in testing mode
        if not self.config.TESTING_MODE:
            self.genera_client.authenticate()
        
        # Define parameter space
        parameters = self.get_parameter_config()
        
        # Run optimization
        best_parameters, best_values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=lambda p: self.run_assay(
                p["pH"], p["enzyme_conc"], p["incubation_time"]
            ),
            objective_name="assay_score",
            total_trials=self.config.TOTAL_TRIALS,
            minimize=False  # We want to maximize the assay score
        )
        
        # Save results
        timestamp = self.save_results(experiment, model, best_parameters, best_values)
        
        # Plot results
        self.plot_optimization_results(experiment, f"bo_plot_{timestamp}")
        
        # Print final results
        print(f"\nOptimization completed successfully!")
        print(f"Best Parameters: {best_parameters}")
        print(f"Best Value: {best_values}")
        
        return best_parameters, best_values, experiment, model


class FlaskApp:
    """Flask application for API endpoints."""
    
    def __init__(self, config: Config):
        self.config = config
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route("/test", methods=["GET"])
        def test():
            return jsonify({"result": "test", "status": "OK"})
        
        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    def run(self):
        """Run the Flask application."""
        self.app.run(host=self.config.FLASK_HOST, port=self.config.FLASK_PORT, debug=False)


def main():
    """Main entry point for the optimization system."""
    try:
        # Initialize configuration
        config = Config()
        
        # Create optimizer
        optimizer = AssayOptimizer(config)
        
        # Run optimization
        best_params, best_values, experiment, model = optimizer.run_optimization()
        
        print("Optimization completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise


if __name__ == "__main__":
    # Uncomment to run Flask server in separate thread
    # config = Config()
    # flask_app = FlaskApp(config)
    # flask_thread = threading.Thread(target=flask_app.run)
    # flask_thread.daemon = True
    # flask_thread.start()
    
    main()