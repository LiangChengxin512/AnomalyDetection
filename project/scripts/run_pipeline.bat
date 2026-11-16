@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0.." || exit /b 2
if defined PYTHON (
  set "PY_EXE=%PYTHON%"
) else (
  set "PY_EXE=python"
)
"%PY_EXE%" "%~dp0reproduce_all.py" %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
