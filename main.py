from flask import Flask, request, jsonify

# python -m flask --app main run --host=0.0.0.0 --port=5000

TOTAL = 0

app = Flask(__name__)

@app.route("/updateParams", methods=["POST"])
def hi():
    global TOTAL
    
    data = request.get_json()
    a = data['a']
    b = data['b']
    result = a + b
    TOTAL += result
    return jsonify({"result": result})

@app.route("/showParams", methods=["GET"])
def print():
    return jsonify({"result": TOTAL})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)