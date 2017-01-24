# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Tells the app what the user gets to see

from app import app


# The index page
@app.route('/')
@app.route('/index')
def index():
    return "Hello World!"


# The prediction ap
@app.route("/api/predict", methods=['POST'])
def predict():
    pass
