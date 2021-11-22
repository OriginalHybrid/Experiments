
from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

from flasgger import swag_from

@app.route('/', methods = ['GET'])
def check():
    return 'Welcome to Flask'

@app.route('/greeting', methods = ['GET'])
@swag_from('swagger_docs/greeting.yml')
def greeting():
    return 'Hello, World!'

@app.route('/post_greeting', methods=['POST'])
@swag_from('swagger_docs/post_greeting.yml')
def post_greeting():
    data = request.get_json()
    name = data['name']
    return 'Hello,' + str(name)

if __name__ == "__main__":
    app.run(debug=True)