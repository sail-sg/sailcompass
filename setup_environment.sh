cd opencompass

pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
pip install pythainlp langid
pip install datasets huggingface_hub
pip uninstall flash-attn -y

cd ..

cd rouge
python setup.py install
cd ..