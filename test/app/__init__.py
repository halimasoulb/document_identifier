from flask import Flask ,render_template


def create_app():
	app = Flask(__name__)

	@app.route('/')
	def homepage():
		return render_template('homepage.html')

	@app.route('/hello/<name>')
	def hello(name):
		return render_template('hellopage.html', name=name)
	
	return app