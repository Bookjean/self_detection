# Self Detection MLP Training Guide (Revised)

## 1. Objective

This project addresses the **Self Detection problem** of single-ended capacitive proximity sensors mounted on a robot manipulator.

**Self Detection** refers to sensor output variations caused solely by the robot’s own posture, even when **no external object is present**.

The objective is to:

- Use **all robot joint angles** as inputs
- Predict the **posture-dependent self baseline** of capacitive sensors
- Apply **offset-only compensation**
- Preserve all external-object-induced signals

This model is explicitly **not** designed to detect objects, but to **remove self-induced baseline effects**.

---

## 2. Problem Definition

### Measurement Model

\[
y(t) = y_{\text{ext}}(t) + b(q(t)) + \text{noise}
\]

where:

- \(y(t)\): measured sensor output  
- \(y_{\text{ext}}(t)\): signal caused by external objects  
- \(b(q)\): posture-dependent self baseline  
- noise: measurement noise

### Learning Target

\[
\hat b(q) \approx b(q)
\]

Compensation is applied as:

\[
y_{\text{corrected}}(t) = y(t) - \hat b(q(t))
\]

---

## 3. Design Decisions

### 3.1 Input Selection

**Inputs**
- Joint angles: `j1, j2, j3, j4, j5, j6`

**Input dimension:** 6

**Rationale**

Self detection in capacitive sensors is primarily driven by **geometric configuration**:
- Link-to-link capacitive coupling
- Ground reference redistribution
- Structural proximity effects

Joint velocities are intentionally excluded to:
- Reduce noise amplification
- Avoid numerical differentiation
- Enforce a strictly **static baseline model**

---

## 4. Data Format

```
timestamp,
j1, j2, j3, j4, j5, j6,
prox1, prox2, prox3, prox4,
raw1, raw2, raw3, raw4,
tof1, tof2, tof3, tof4
```

**Training targets**
- `raw1, raw2, raw3, raw4`

---

## 5. Training Data Assumptions

- Data collected with **no external objects**
- No gating applied
- Model learns **self baseline only**

⚠️ External objects in training data may cause unsafe over-compensation.

---

## 6. Output Standardization

For each sensor channel:

\[
y_{\text{norm}} = \frac{y - \mu}{\sigma}
\]

Runtime de-normalization:

\[
\hat y = \hat y_{\text{norm}} \cdot \sigma + \mu
\]

---

## 7. Offset-Only Self Detection Model

- Offset-only modeling
- No gain compensation
- No temporal dynamics

Preserves external signals and safety margins.

---

## 8. MLP Architecture

**Type:** MIMO

```
Input (6)
 → Dense(32) + ReLU
 → Dense(32) + ReLU
 → Dense(32) + ReLU
 → Dense(4)
```

---

## 9. Training and Evaluation

- Time-based split (e.g., 70/30)
- Metrics:
  - Standard deviation reduction
  - Peak-to-peak reduction

---

## 10. Processing Pipeline

1. Load CSV
2. Extract `j1–j6`
3. Extract `raw1–raw4`
4. Standardize outputs
5. Train MLP
6. De-normalize predictions
7. Apply compensation

---

## 11. Scope and Limitations

Handles:
- Static posture-dependent baseline

Does not handle:
- Hysteresis
- Dynamic grounding effects
- High-speed motion artifacts

---

## Summary

This document defines a **static posture-based self detection compensation framework** that is conservative, interpretable, and suitable as a Stage-1 baseline remover.