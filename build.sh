# exit on error
set -o errexit

pip3 install -r requirements.txt
python manage.py collectstatic --no-input