from flask import Flask, request, jsonify
import main

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def api():
    question = request.json['question']
    answer = main.answer_question(question)

    # Get the client's IP address
    client_ip = "http://192.168.1.17"

    response = {
        'answer': answer,
        'ip_address': client_ip
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='192.168.1.17', port=5000)
