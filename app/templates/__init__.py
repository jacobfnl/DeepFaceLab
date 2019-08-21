from flask import Flask
app = Flask(__name__,
 static_folder = './public',
 template_folder="./static")

from templates.root.views import root_blueprint

# register the blueprint
app.register_blueprint(root_blueprint)