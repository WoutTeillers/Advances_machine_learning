# Advanced_machine_learning

In this study, we used different machine learning methods to tackle the three body problem.
We simplified the problem to include only the x, y positions resulting in 12-Dimensional vectors (x1, x2, x3, y1, y2, y3, vx1, vx2, vx3, vy1, vy2, vy3)

## Approaches:

- Stateless LSTM
- LSTM with forecast horizon
- Stateful LSTM
- MLP with 12-Dimensional output
- MLP with 6-Dimensional output and numerically calculated velocoties

# Project Setup and Usage

## 1. Create a virtual environment

```
python -m venv venv
```

## 2. Activate virtual environment

**Windows (PowerShell):**

```
venv\Scripts\Activate
```

**MacOS / Linux**

```
source venv/bin/activate
```

## 3. Install Dependencies

```
pip install -r requirements.txt
```

## 4. Run Application

- Stateless LSTM

```
python main_stateless.py 1
```

- LSTM with forecast horizon
  Make sure lag > 1

```
python main_stateless.py [lag]
```

- Stateful LSTM

```
python main.py
```

- MLP with 12-Dimensional output

```
python main_mlp.py
```

- MLP with 6-Dimensional output and numerically calculated velocoties

```
python main_mlp_6d.py
```
