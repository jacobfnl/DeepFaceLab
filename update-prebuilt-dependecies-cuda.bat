@echo off
call "..\setenv.bat"

Set git=".\bin\git-cmd"

python -c "from pip._internal import pep425tags; print('Supported:', pep425tags.get_supported(), '\n')"

pip install .\bin\eos_py-1.1.2-cp36-cp36m-win_amd64.whl

pause