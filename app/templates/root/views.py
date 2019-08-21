from flask import render_template, Blueprint
root_blueprint = Blueprint('root', __name__)
@root_blueprint.route('/')
@root_blueprint.route('/root')
def index():
 return render_template("index.html")