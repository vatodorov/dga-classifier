#########################################################################
##
##   All right reserved (c)2020 - Valentin Todorov
##
##   Purpose: Analyze the model results
##
#########################################################################


# =============== Environment variables =============== #

# App name
APP_NAME=mc_dga_classifier

# Folder with the deployment package for the app
APP_DEPLOY_LOC=/tmp/${APP_NAME}

# Path to the Python virtual env for the app
VIRTUAL_ENV_PATH=/opt/venvs/dga-app

# Create a config file for the app
APP_PORT=5001
APP_CONFIG_DIR=/opt/${APP_NAME}
LOG_LOCATION=/opt/${APP_NAME}/logs
LOG_FILE=${APP_NAME}.log

# ===================================================== #


# Enable the virtual environment for the Flask app
source ${VIRTUAL_ENV_PATH}/bin/activate

# Install gunicorn which will be used to run the flask server
GUNI_TEST=$(pip list | grep gunicorn | awk '{print $2""$3}')
if [ ${GUNI_TEST} ]; then
  echo ""
  echo "Gunicorn version ${GUNI_TEST} is installed. Continuing..."
else
  echo "Gunicorn is not installed on this machine, and will be installed."
  pip install gunicorn
fi

sleep 3

echo -e "\nCreating the server config directory ${APP_CONFIG_DIR}"
rm -rf ${APP_CONFIG_DIR}
mkdir ${APP_CONFIG_DIR}

echo "Creating the log location directory ${LOG_LOCATION}"
rm -rf ${LOG_LOCATION}
mkdir ${LOG_LOCATION}

sleep 3

# User input for the config
echo -e "\nPlease provide the following server configuration values..."
read -p "Enter the main path to the models without the analysis date: " MODEL_LOCATION
read -p "Enter the analysis date for the model: " ANALYSIS_DATE
read -p "Enter name of the model: " MODEL_NAME
read -p "Enter the cutoff for the DGA score: " DGA_SCORE_CUTOFF
read -p "Should the app run single- or multi-threaded? " THREADS_MODE


# Configure the app
echo -e "\nStarting the configuration of the DGA Classifier server"
echo "The app will run on port ${APP_PORT}"

sleep 1

cat << EOF > "${APP_CONFIG_DIR}/config.yaml"
data
  model_location: ${MODEL_LOCATION}
  analysis_date: ${ANALYSIS_DATE}
  model_name: ${MODEL_NAME}
  dga_score_cutoff: ${DGA_SCORE_CUTOFF}
  app_host: 0.0.0.0
  app_port: ${APP_PORT}
  app_threads_mode: ${THREADS_MODE}
EOF


# Create a file to store the ProxyPass config for Apache
# If I'm only running the Flask app this should be ok
# However, if we are also running a website this should be modified
echo "Adding a directive to the Apache config file..."

sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod proxy_balancer
sudo a2enmod lbmethod_byrequests
sudo systemctl restart apache2

cat << EOF > "/etc/apache2/sites-available/000-default.conf"
<VirtualHost *:80>
        # The ServerName directive sets the request scheme, hostname and port that
        # the server uses to identify itself. This is used when creating
        # redirection URLs. In the context of virtual hosts, the ServerName
        # specifies what hostname must appear in the request's Host: header to
        # match this virtual host. For the default virtual host (this file) this
        # value is not decisive as it is used as a last resort host regardless.
        # However, you must set it for any further virtual host explicitly.
        #ServerName www.example.com

        ServerAdmin webmaster@localhost
        DocumentRoot /var/www/html

        # Available loglevels: trace8, ..., trace1, debug, info, notice, warn,
        # error, crit, alert, emerg.
        # It is also possible to configure the loglevel for particular
        # modules, e.g.
        #LogLevel info ssl:warn

        ErrorLog ${APACHE_LOG_DIR}/error.log
        CustomLog ${APACHE_LOG_DIR}/access.log combined

        # For most configuration files from conf-available/, which are
        # enabled or disabled at a global level, it is possible to
        # include a line for only one particular virtual host. For example the
        # following line enables the CGI configuration for this host only
        # after it has been globally disabled with "a2disconf".

        #Include conf-available/serve-cgi-bin.conf

        ProxyPreserveHost On
        ProxyPass / http://localhost:5001/
        ProxyPassReverse / http://localhost:5001/

</VirtualHost>
EOF

systemctl restart apache2


echo -e "\nInstalling the app server..."
cd /tmp/ && pip install ./manticore


# Run the app as a service
echo -e "\nSetting up the DGA Classifier to run as a service"
LOG_LEVEL=DEBUG
LOG_LOCATION_FILE=${LOG_LOCATION}/${LOG_FILE}

touch "/etc/systemd/system/${APP_NAME}.service"
cat << EOF > "/etc/systemd/system/${APP_NAME}.service"
[Unit]
Description=Flask app server that runs the DGA Classifier scoring API

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/opt/venvs/dga-app/bin/gunicorn mc_dga_classifier.app:app --bind 0.0.0.0:5001 --pythonpath /opt/venvs/dga-app/bin/python --log-level info --error-logfile /var/log/mc_dga_classifier.log
Restart=always
TimeoutSec=10

[Install]
WantedBy=multi-user.target
EOF

sleep 3

# Enable the service
echo -e "\nOpen port ${APP_PORT} in the firewall"
ufw allow ${APP_PORT}/tcp
ufw enable

echo "The following are the open ports on this instance"
ufw status verbose

sleep 5

echo -e "\nRestarting the daemon and enabling the server to start after a reboot"
chmod 664 "/etc/systemd/system/${APP_NAME}.service"
systemctl daemon-reload
systemctl enable ${APP_NAME}.service
systemctl restart ${APP_NAME}

sleep 5

echo "Restarting Apache. Test if it has restarted successfully"
systemctl reload apache2
systemctl status apache2

