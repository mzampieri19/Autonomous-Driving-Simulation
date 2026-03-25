# Autonomous-Driving-Simulation

How to run project locally: 

## 1 Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2 Train the agent

```bash
python3 train.py
```

This saves training artifacts in `outputs/training/` (including the trained agent and reward history).

## 3 Evaluate the trained agent

```bash
python3 evaluate.py
```

This compares the trained policy vs random baseline and saves a report image to `outputs/evaluation/evaluation.png`.

## 4 (Optional) Hyperparameter optimization

```bash
python3 optimize.py
```

Results are written to `outputs/optimization/`. This can take a long time because it runs many parameter combinations.

## Notes

- Run all commands from the project root.
- Keep your virtual environment activated while running scripts.
