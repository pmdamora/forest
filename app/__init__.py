# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Initiates the app

from flask import Flask

# Initialize the app from configuration file
app = Flask(__name__)
app.config.from_object('app.config.settings')

# Import views
from app.pages import views
