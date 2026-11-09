# One-time setup and smoke test

1. Start the backend using the commands in `README.md`.
2. Confirm `http://localhost:8000/docs` opens.
3. Start the frontend in a second terminal using the commands in `README.md`.
4. Open the Vite URL printed in the terminal.
5. Choose a dataset, select input features (use **Select All** when appropriate), choose the target label, and train a model.
6. Confirm the metrics appear. Then test prediction/forward propagation using the fields shown by the playground.

For DS3, `early_menarche` is the target label. It represents the retrospective classification threshold described in `datasets/README.md`; it is not a clinical diagnosis or a prediction of a child's future puberty timing.
