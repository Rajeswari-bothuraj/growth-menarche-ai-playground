# Growth & Menarche AI Playground

Interactive React + FastAPI + PyTorch playground for selecting features, choosing a target, training models, viewing metrics, and testing predictions.

## Included datasets

See `datasets/README.md` for the three prepared IHDS-II-derived CSV files (DS1, DS3, DS4) and their limitations. For the DS3 early-menarche task, the target is `early_menarche`: 1 = reported menarche before age 12; 0 = age 12 or older. `EW6` is current respondent age, not menarche age.

## Run on Chromebook/Linux

Open Terminal and navigate into the extracted project folder, then:

### Backend terminal

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend: http://localhost:8000
API docs: http://localhost:8000/docs

### Frontend terminal (open a second terminal)

From the project root:

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open the local Vite URL shown in the terminal (normally http://localhost:5173).

## Public deployment

GitHub stores the source code but does not run the FastAPI/PyTorch backend. Deploy the backend to a Python-capable host and the frontend to a static host (or deploy both to a host that supports both). Set the frontend API base URL to the deployed backend URL before publishing. Do not put private data, secrets, or API keys in the repository.
