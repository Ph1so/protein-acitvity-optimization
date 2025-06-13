from flask import Flask, request, jsonify
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.acquisition import gaussian_ei

# python -m flask --app main run --host=0.0.0.0 --port=5000

app = Flask(__name__)

# Global variables for Bayesian optimization
# Define parameter search space for biological/chemical parameters
PARAM_SPACE = [
    Real(20, 100, name='temperature'),        # Temperature between 20°C and 100°C
    Real(0.5, 24, name='incubation_time'),    # Incubation time between 0.5 and 24 hours
    Real(10, 1000, name='volume')             # Volume between 10 and 1000 mL
]

# Store optimization history
X_OBSERVED = []  # Parameter combinations tried
Y_OBSERVED = []  # Results (we'll minimize this, so lower is better)

# Current best parameters (initialized to middle of ranges)
CURRENT_PARAMS = {
    'temperature': 37.0,      # Default temperature
    'incubation_time': 6.0,   # Default 6 hours
    'volume': 100.0           # Default 100 mL
}

# curl -X POST -H "Content-Type: application/json" \
#   -d '{"result": 0.85}' \
#   http://127.0.0.1:5000/updateParams
@app.route("/updateParams", methods=["POST"])
def update_params():
    global X_OBSERVED, Y_OBSERVED, CURRENT_PARAMS
    
    try:
        data = request.get_json()
        
        # Get the result from training (assuming lower is better, e.g., loss)
        # If higher is better (e.g., accuracy), negate it: result = -data['result']
        result = data['result']
        
        # Add current params and result to history
        current_x = [
            CURRENT_PARAMS['temperature'],
            CURRENT_PARAMS['incubation_time'], 
            CURRENT_PARAMS['volume']
        ]
        
        X_OBSERVED.append(current_x)
        Y_OBSERVED.append(result)
        
        print(f"Received result: {result} for params: {CURRENT_PARAMS}")
        
        # If we have enough data points, use Bayesian optimization
        if len(X_OBSERVED) >= 2:
            # Perform Bayesian optimization to suggest next parameters
            result_gp = gp_minimize(
                func=lambda x: 0,  # Dummy function (we provide data directly)
                dimensions=PARAM_SPACE,
                x0=X_OBSERVED,
                y0=Y_OBSERVED,
                n_calls=0,  # Don't call the function, just use existing data
                acq_func='EI',  # Expected Improvement
                random_state=42
            )
            
            # Get next suggested parameters
            next_params = result_gp.ask()
            
            # Update global parameters
            CURRENT_PARAMS = {
                'temperature': float(next_params[0]),
                'incubation_time': float(next_params[1]),
                'volume': float(next_params[2])
            }
        else:
            # For first few iterations, use random exploration
            CURRENT_PARAMS = {
                'temperature': float(np.random.uniform(20, 100)),
                'incubation_time': float(np.random.uniform(0.5, 24)),
                'volume': float(np.random.uniform(10, 1000))
            }
        
        print(f"New suggested params: {CURRENT_PARAMS}")
        
        return jsonify({
            "status": "success",
            "new_params": CURRENT_PARAMS,
            "iteration": len(X_OBSERVED),
            "best_result_so_far": float(min(Y_OBSERVED)) if Y_OBSERVED else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/getParams", methods=["GET"])
def get_params():
    """Get current parameters for training"""
    return jsonify({
        "params": CURRENT_PARAMS,
        "iteration": len(X_OBSERVED)
    })

@app.route("/reset", methods=["POST"])
def reset_optimization():
    """Reset the optimization history"""
    global X_OBSERVED, Y_OBSERVED, CURRENT_PARAMS
    
    X_OBSERVED = []
    Y_OBSERVED = []
    CURRENT_PARAMS = {
        'temperature': 37.0,
        'incubation_time': 6.0,
        'volume': 100.0
    }
    
    return jsonify({"status": "reset_complete"})

@app.route("/history", methods=["GET"])
def get_history():
    """Get optimization history"""
    history = []
    for i, (x, y) in enumerate(zip(X_OBSERVED, Y_OBSERVED)):
        history.append({
            "iteration": i + 1,
            "params": {
                'temperature': x[0],
                'incubation_time': x[1],
                'volume': x[2]
            },
            "result": y
        })
    
    return jsonify({
        "history": history,
        "best_result": float(min(Y_OBSERVED)) if Y_OBSERVED else None,
        "total_iterations": len(X_OBSERVED)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)