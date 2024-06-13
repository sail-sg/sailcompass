cd opencompass

pip install -e .
pip install -r requirements.txt
pip install pythainlp langid
pip install datasets huggingface_hub
pip uninstall flash-attn

cd ..