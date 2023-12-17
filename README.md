# AskGottfired

## Create Virtual Environment
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## Install Requirements

Download Windows SDK-Version for your pc
Install visual studio 2022 with c++ dev
Install mingw 




```
$env:CMAKE_ARGS = "-DLLAMA_OPENBLAS=on -DCMAKE_C_COMPILER=C:/MinGW/bin/gcc.exe -DCMAKE_CXX_COMPILER=C:/MinGW/bin/g++.exe"
pip install -r requirements.txt
```


## Download Model
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF



## Run
```
python -m streamlit run .\Gottfried_v2.py
```