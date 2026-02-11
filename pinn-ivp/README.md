# Neural Network ODE Solver (PINN-style)

This project implements a neural network–based solver for ordinary differential equations using automatic differentiation in JAX. The network is trained to satisfy a differential equation and boundary condition directly, rather than fitting labeled data.

The example problem solved is:

y'(x) − 2x y(x) = 0  
with initial condition y(0) = 1

The analytical solution is:

y(x) = exp(x²)

The neural network approximation is trained to recover this solution.

Code reference: :contentReference[oaicite:0]{index=0}

📄 [Final Report (PDF)](report.pdf)  
🎞️ [Presentation Slides](slides.pdf)  
🎥 [Presentation Video](https://youtu.be/NBF0fpon3IY)

---

## Method

This is a simple physics-informed neural network (PINN) approach:

- A neural network represents y(x)
- Automatic differentiation computes derivatives
- A loss function enforces the differential equation
- A boundary condition penalty is added
- Gradient descent trains the network

No labeled training data is required. The model learns by minimizing the residual of the governing equation.

---

## Features

- JAX-based autodifferentiation
- Momentum gradient descent
- Vectorized evaluation with `vmap`
- JIT-compiled loss and gradients
- Real-time visualization of convergence
- Comparison with analytical solution

The framework is easily extendable to other ODEs and boundary value problems.

---

## Output

The program produces:

- solution plots vs analytical solution
- convergence behavior during training
- loss evolution

These visualize how the neural network learns the differential equation.

---

## Purpose

This project demonstrates:

- neural differential equation solvers
- physics-informed learning
- automatic differentiation in JAX
- PINN-style training without labeled data

It serves as a minimal research prototype for scientific machine learning experiments.

---

## Authors

Melissa Butler, Nick Baumgartner, Paul Sansah  
University of Wyoming

