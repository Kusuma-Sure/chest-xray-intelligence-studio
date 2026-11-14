# Windows quick start

1. Extract the downloaded ZIP. Open CMD in the inner `chexpert_capstone` folder (the folder containing `backend`, `frontend`, `app.py` and `requirements.txt`).
2. Run:

```cmd
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

3. Start the local API and frontend:

```cmd
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

4. Browse to `http://127.0.0.1:8000`. Keep the CMD window open.
5. Check model checkpoint status with a second CMD window in the same project folder (activate `.venv` first):

```cmd
python scripts\check_models.py
```

The dashboard is runnable before training, but prediction is enabled only after a real compatible checkpoint is copied into `checkpoints`. Use `docs/CheXpert_Capstone_Training_Colab.ipynb` in Google Colab to train the models. The ZIP itself contains source code, not the CheXpert dataset or trained weights.
