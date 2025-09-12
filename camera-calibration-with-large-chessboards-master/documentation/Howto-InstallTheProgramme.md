# Howto - Install the programme

To install the camera-calibration-with-large-chessboards tool you need to do the following steps. First clone the git repository in an appropriate location.
```
:~/Desktop$ git clone git@source.coderefinery.org:sdu-uas-center/perception/camera-calibration-with-large-chessboards.git
Cloning into 'camera-calibration-with-large-chessboards'...
remote: Enumerating objects: 237, done.
remote: Counting objects: 100% (33/33), done.
remote: Compressing objects: 100% (24/24), done.
remote: Total 237 (delta 17), reused 18 (delta 9), pack-reused 204
Receiving objects: 100% (237/237), 793.67 KiB | 6.10 MiB/s, done.
Resolving deltas: 100% (122/122), done.
```

Then enter the git repository and install the python requirements using pipenv.
```
:~/Desktop$ cd camera-calibration-with-large-chessboards/
:~/Desktop/camera-calibration-with-large-chessboards/input$ pipenv install
Creating a virtualenv for this project...
Pipfile: /home/hemi/Desktop/camera-calibration-with-large-chessboards/Pipfile
Using /usr/bin/python3.8 (3.8.10) to create virtualenv...
⠼ Creating virtual environment...created virtual environment CPython3.8.10.final.0-64 in 238ms
  creator CPython3Posix(dest=/home/hemi/.local/share/virtualenvs/camera-calibration-with-large-chessboards-UcwRDchb, clear=False, no_vcs_ignore=False, global=False)
  seeder FromAppData(download=False, pip=bundle, setuptools=bundle, wheel=bundle, via=copy, app_data_dir=/home/hemi/.local/share/virtualenv)
    added seed packages: pip==22.2.2, setuptools==65.4.1, wheel==0.37.1
  activators BashActivator,CShellActivator,FishActivator,PowerShellActivator,PythonActivator,XonshActivator

✔ Successfully created virtual environment! 
Virtualenv location: /home/hemi/.local/share/virtualenvs/camera-calibration-with-large-chessboards-UcwRDchb
Pipfile.lock not found, creating...
Locking [dev-packages] dependencies...
Locking [packages] dependencies...
Building requirements...
Resolving dependencies...
✔ Success! 
Updated Pipfile.lock (659943)!
Installing dependencies from Pipfile.lock (659943)...
  🐍   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 13/13 — 00:00:04
To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
```

