#########################################################################
##
##   All right reserved (c)2020 - Valentin Todorov
##
##   Purpose: Analyze the model results
##
#########################################################################

# Install gunicorn which will be used to run the flask server
pip install gunicorn

echo "Installing the app server..."
pip install dga_classifier_server


# Create a config file for the app
APP_PORT=5001
APP_CONFIG_DIR=/opt/dga-classifier-server
LOG_LOCATION=/opt/dga-classifier-server/logs
LOG_FILE=dga_classifier_server.log

echo "Createing the server config directory ${APP_CONFIG_DIR}"
mkdir ${APP_CONFIG_DIR}

echo "Creating the log location directory ${LOG_LOCATION}"
mkdir ${LOG_LOCATION}


# User input for the config
echo "Please provide the following server configuration values..."
read -p "Enter the main path to the models without the analysis date: " MODEL_LOCATION
read -p "Enter the analysis date for the model: " ANALYSIS_DATE
read -p "Enter name of the model: " MODEL_NAME
read -p "Enter the cutoff for the DGA score: " DGA_SCORE_CUTOFF


# Configure the app
echo "Starting the configuration of the DGA Classifier server"
echo "The app will run on port ${APP_PORT}"

cat << EOF > "${APP_CONFIG_DIR}/config.yaml"
---
model_location=${MODEL_LOCATION}
analysis_date=${ANALYSIS_DATE}
model_name=${MODEL_NAME}
dga_score_cutoff=${DGA_SCORE_CUTOFF}
app_host='0.0.0.0'
app_port=${APP_PORT}
EOF


# Apache directive
# DO I NEED THIS PART????
echo "Adding a directive to the Apache file..."
<Location /manticore>
    ProxyPreserveHost On
    ProxyPass http://localhost:${APP_PORT}/
    ProxyPassReverse http://localhost:${APP_PORT}/
</Location>
EOF


# Run the app as a service
echo "Setting up the DGA Classifier to run as a service"
LOG_LEVEL=DEBUG
LOG_LOCATION_FILE="${LOG_LOCATION}/${LOG_FILE}"

cat << EOF > "/etc/systemd/system/dga_classifier.service"
[Unit]
Description=Flask app server that runs the DGA Classifier scoring API

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/bin/gunicorn dga_classifier_server:app --bind 0.0.0.0:${APP_PORT} --config python:dga_classifier_server --log-level ${LOG_LEVEL} --log-location ${LOG_LOCATION_FILE}
Restart=always
TimeoutSec=10

[Install]
WantedBy=multi-user.target
EOF


# Enable the service
echo "Restarting the daemon and enabling the server to start after a reboot"
chmod 664 /etc/systemd/dga_classifier.service
systemctl daemon-reload
systemctl enable dga_classifier.service
systemctl restart dga_classifier

echo "Restarting httpd. Test if it has restarted successfully"
systemctl restart httpd
systemctl status httpd

