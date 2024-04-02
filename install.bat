@ECHO OFF
echo Creating VENV...
python -m venv venv
call venv\Scripts\activate
echo Installing requirements...
pip install -r requirements.txt
echo Installation finished. You can run the app now.
pause
exit
