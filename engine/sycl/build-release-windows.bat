@echo off
rem Build the portable Intel engine and vision encoder, then package the standard Windows engine zip.
rem Needs oneAPI DPC++/C++, oneMKL, Visual Studio C++ tools, Python, CMake, Ninja and Git.
setlocal
for %%I in ("%~dp0..\..") do set "STRATA_SRC=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%STRATA_SRC%\build-sycl-release"
if not defined VISION_BUILD_DIR set "VISION_BUILD_DIR=%STRATA_SRC%\build-vision-sycl-release"
if not defined DIST_DIR set "DIST_DIR=%STRATA_SRC%\dist"
set "STRATA_ONEAPI_INIT=%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat"
if defined ONEAPI_ROOT if exist "%ONEAPI_ROOT%\setvars.bat" set "STRATA_ONEAPI_INIT=%ONEAPI_ROOT%\setvars.bat"
if not exist "%STRATA_ONEAPI_INIT%" (
    echo Intel oneAPI setvars.bat not found.
    exit /b 1
)
call "%STRATA_ONEAPI_INIT%" intel64
if not "%errorlevel%"=="0" exit /b %errorlevel%
set "STRATA_RELEASE_PY="
if exist "%STRATA_SRC%\terminal\.venv\Scripts\python.exe" set "STRATA_RELEASE_PY=%STRATA_SRC%\terminal\.venv\Scripts\python.exe"
if not defined STRATA_RELEASE_PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined STRATA_RELEASE_PY set "STRATA_RELEASE_PY=%%P"
if not defined STRATA_RELEASE_PY (
    echo Python 3.10+ is needed.
    exit /b 1
)
cmake -S "%STRATA_SRC%\engine\sycl" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_PORTABLE=ON -DSTRATA_SYCL_PARITY=OFF
if not "%errorlevel%"=="0" exit /b %errorlevel%
cmake --build "%BUILD_DIR%" --target strata strata-device --parallel 4
if not "%errorlevel%"=="0" exit /b %errorlevel%
cmake -S "%STRATA_SRC%\terminal\tools\vision" -B "%VISION_BUILD_DIR%" -G Ninja ^
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_PORTABLE=ON ^
    -DSTRATA_VISION_CUDA=OFF -DSTRATA_VISION_SYCL=ON
if not "%errorlevel%"=="0" exit /b %errorlevel%
cmake --build "%VISION_BUILD_DIR%" --target strata-vision --parallel 4
if not "%errorlevel%"=="0" exit /b %errorlevel%
"%STRATA_RELEASE_PY%" "%STRATA_SRC%\engine\sycl\tools\package_windows.py" ^
    --build "%BUILD_DIR%" --vision-build "%VISION_BUILD_DIR%" --out "%DIST_DIR%"
exit /b %errorlevel%
