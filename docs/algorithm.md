# The RKF7(8) Algorithm

A detailed explanation of the Runge-Kutta-Fehlberg 7(8) method as implemented in this crate.

---

## 1. Introduction

The Runge-Kutta-Fehlberg 7(8) method (RKF78) is a high-order embedded pair for solving initial value problems of the form:

$$\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$$

It was developed by Erwin Fehlberg at NASA's Marshall Space Flight Center and published in 1968 as NASA Technical Report R-287: *"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control"*. The method provides an **8th-order accurate solution** with a **7th-order embedded estimate** for adaptive step-size control, using **13 function evaluations** (stages) per step.

RKF78 has been widely adopted in production astrodynamics software for spacecraft trajectory propagation, including NASA's own codes, because:

- The high order means large step sizes can be taken while maintaining accuracy
- The embedded error estimate enables automatic step-size adaptation
- Decades of flight heritage provide confidence in the method's reliability

This crate implements the coefficients from **Table X (pages 64-65)** of NASA TR R-287.

---

## 2. Runge-Kutta Methods Background

An **explicit Runge-Kutta method** with $s$ stages advances the solution from $y_n$ at time $t_n$ to $y_{n+1}$ at time $t_n + h$ by:

1. Computing $s$ intermediate slopes (stages):

$$k_i = f\!\left(t_n + c_i h,\; y_n + h \sum_{j=0}^{i-1} a_{ij}\, k_j\right), \quad i = 0, 1, \ldots, s-1$$

2. Combining them into the solution update:

$$y_{n+1} = y_n + h \sum_{i=0}^{s-1} b_i\, k_i$$

The coefficients $c_i$, $a_{ij}$, and $b_i$ define the method and are traditionally organized in a **Butcher tableau**:

$$\begin{array}{c|c}
\mathbf{c} & A \\
\hline
& \mathbf{b}^T
\end{array}$$

where $A$ is a strictly lower-triangular matrix (for explicit methods), $\mathbf{c}$ is the vector of node coefficients, and $\mathbf{b}$ is the vector of weights.

### Embedded Pairs

An **embedded pair** provides two solutions of different orders from the same set of stages. For RKF7(8), both a 7th-order solution $\hat{y}_{n+1}$ and an 8th-order solution $y_{n+1}$ are computed from the same 13 stage evaluations, using different weight vectors $\hat{\mathbf{b}}$ and $\mathbf{b}$. The difference between them estimates the local truncation error without any extra function evaluations.

---

## 3. The RKF7(8) Pair

The method uses **13 stages** ($s = 13$). Stages 0-10 produce the 8th-order solution. Stages 11 and 12 are additional evaluations used only by the 7th-order error estimate.

### Node Coefficients

The node coefficients $c_i$ determine at which points within the step the derivative is evaluated:

| $i$ | $c_i$ | Decimal |
|-----|--------|---------|
| 0 | $0$ | 0.0 |
| 1 | $2/27$ | 0.074074... |
| 2 | $1/9$ | 0.111111... |
| 3 | $1/6$ | 0.166667... |
| 4 | $5/12$ | 0.416667... |
| 5 | $1/2$ | 0.5 |
| 6 | $5/6$ | 0.833333... |
| 7 | $1/6$ | 0.166667... |
| 8 | $2/3$ | 0.666667... |
| 9 | $1/3$ | 0.333333... |
| 10 | $1$ | 1.0 |
| 11 | $0$ | 0.0 |
| 12 | $1$ | 1.0 |

Note the unusual structure: stages 11 and 12 re-evaluate at $t_n$ and $t_n + h$ respectively, but with different state arguments (they depend on all previous stages). This is specific to Fehlberg's construction for the error estimate.

### Row-Sum Condition

The Butcher tableau satisfies the consistency condition:

$$\sum_{j=0}^{i-1} a_{ij} = c_i \quad \text{for all } i$$

This is verified by the `test_row_sum_condition` test in `src/coefficients.rs`.

### 8th-Order Weights

The 8th-order solution uses weights $b_i$:

| $i$ | $b_i$ |
|-----|--------|
| 0 | $41/840$ |
| 1-4 | $0$ |
| 5 | $34/105$ |
| 6 | $9/35$ |
| 7 | $9/35$ |
| 8 | $9/280$ |
| 9 | $9/280$ |
| 10 | $41/840$ |
| 11-12 | $0$ |

These weights sum to 1 (verified by `test_weights_sum_to_one`). Notice the symmetry: $b_0 = b_{10} = 41/840$, $b_6 = b_7 = 9/35$, and $b_8 = b_9 = 9/280$.

### 7th-Order Weights

The 7th-order embedded solution uses weights $\hat{b}_i$:

| $i$ | $\hat{b}_i$ |
|-----|--------|
| 0-4 | $0$ |
| 5 | $34/105$ |
| 6 | $9/35$ |
| 7 | $9/35$ |
| 8 | $9/280$ |
| 9 | $9/280$ |
| 10 | $0$ |
| 11 | $41/840$ |
| 12 | $41/840$ |

These also sum to 1. The 7th-order solution shares weights with the 8th-order solution for stages 5-9, but differs for stages 0, 10, 11, and 12.

### RK Matrix

The full $13 \times 12$ RK matrix $A$ is stored in `src/coefficients.rs` as the constant `A`. Each row $i$ has at most $i$ nonzero entries. The complete matrix is too large to reproduce here but follows exactly from NASA TR R-287 Table X ($\beta$ values). See `src/coefficients.rs` for the exact rational coefficients.

---

## 4. Error Estimation

The local truncation error is estimated from the difference between the 8th-order and 7th-order solutions:

$$\mathrm{TE} = y_{n+1} - \hat{y}_{n+1} = h \sum_{i=0}^{12} (b_i - \hat{b}_i)\, k_i$$

### Sparse Error Weights

A key efficiency property: the error weight vector $\mathbf{e} = \mathbf{b} - \hat{\mathbf{b}}$ has only **4 nonzero entries** out of 13:

| $i$ | $b_i - \hat{b}_i$ |
|-----|---------------------|
| 0 | $+41/840$ |
| 10 | $+41/840$ |
| 11 | $-41/840$ |
| 12 | $-41/840$ |

All other entries are zero because stages 5-9 have identical weights in both solutions. This means the truncation error simplifies to:

$$\mathrm{TE} = \frac{41}{840}\, h\, (k_0 + k_{10} - k_{11} - k_{12})$$

This is the formula given directly in NASA TR R-287. It also verifies that the error weights sum to zero ($41/840 + 41/840 - 41/840 - 41/840 = 0$), which is a necessary condition for the error estimate to vanish for constant solutions.

---

## 5. Adaptive Step-Size Control

### Error Norm

The normalized error is computed using the **infinity norm** with mixed absolute/relative scaling:

$$\epsilon = \max_{i} \frac{|\mathrm{TE}_i|}{a_i + r_i\, |y_i|}$$

where:
- $\mathrm{TE}_i$ is the truncation error in component $i$
- $a_i$ is the absolute tolerance for component $i$
- $r_i$ is the relative tolerance for component $i$
- $y_i$ is the 8th-order solution value for component $i$

The step is **accepted** if $\epsilon \leq 1$ and **rejected** if $\epsilon > 1$.

### I-Controller

The next step size is computed using a standard I-controller (integral controller):

$$h_{\text{new}} = h \cdot \sigma \cdot \epsilon^{-1/p}$$

where:
- $\sigma = 0.9$ is the safety factor (conservative undershoot)
- $p = 8$ is the exponent (matching the error estimate order plus one)
- $\epsilon$ is the normalized error from above

The safety factor ensures that the new step size doesn't overshoot the tolerance threshold, reducing the number of rejected steps.

### Growth Bounds

The step-size adjustment factor is clamped to prevent extreme changes:

$$h_{\text{new}} = h \cdot \text{clamp}\!\left(\sigma \cdot \epsilon^{-1/8},\; 0.2,\; 5.0\right)$$

- **Maximum growth**: $5\times$ per step (prevents overshooting after an easy step)
- **Minimum reduction**: $0.2\times$ per step (prevents catastrophic shrinkage)

### Step Rejection

When a step is rejected ($\epsilon > 1$):
1. The state is not updated
2. A new, smaller step size is computed
3. The step is retried

If the step size falls to `h_min` (default: $10^{-14}$) and the step is still rejected, the integrator returns a `StepSizeTooSmall` error, indicating the problem may be stiff or singular.

---

## 6. Event Detection

The integrator can monitor a user-defined **event function** $g(t, y)$ during integration and detect when it crosses zero.

### Sign-Change Monitoring

After each accepted step, the integrator evaluates $g(t_{n+1}, y_{n+1})$ and checks for a sign change relative to $g(t_n, y_n)$. The check respects the configured `EventDirection`:

- **Rising**: $g_n < 0$ and $g_{n+1} > 0$ (negative to positive)
- **Falling**: $g_n > 0$ and $g_{n+1} < 0$ (positive to negative)
- **Any**: either direction

Special cases: if $g_{n+1} = 0$ exactly, it is treated as a detection. If $g_n = 0$ exactly, it is not treated as a new crossing (to avoid re-detecting the same event).

### Root Finding with Brent's Method

When a sign change is detected in the interval $[t_n, t_{n+1}]$, **Brent's method** is used to locate the precise zero-crossing time. Brent's method combines:

- **Bisection** (guaranteed convergence)
- **Secant method** (fast when well-behaved)
- **Inverse quadratic interpolation** (superlinear convergence near the root)

The algorithm automatically selects the most appropriate technique at each iteration and falls back to bisection when the faster methods would be unreliable. Convergence tolerance defaults to $10^{-12}$ with a maximum of 50 iterations.

During root-finding, the state at intermediate times is obtained by **linear interpolation** between the bracketing states $(t_n, y_n)$ and $(t_{n+1}, y_{n+1})$. This is adequate when adaptive stepping keeps the step sizes reasonable, but could be improved with dense output (Hermite interpolation using the RK stages) in a future version.

### EventAction

When an event is detected:
- **`Stop`** (default): integration terminates and returns the event time and state
- **`Continue`**: the event is recorded in `collected_events` and integration continues past the zero crossing

---

## 7. Implementation Notes

### Const-Generic Design

The integrator is parameterized by state dimension `N` at compile time:

```rust
pub struct Rkf78<const N: usize> { ... }
pub trait OdeSystem<const N: usize> { ... }
```

This allows the compiler to optimize loop bounds, unroll inner loops, and avoid heap allocation for the state vector.

### Pre-Allocated Workspace

The 13 stage vectors are stored as a fixed-size array in the solver struct:

```rust
k: [[f64; N]; STAGES],  // 13 × N workspace
```

No heap allocation occurs during integration (except when collecting events with `EventAction::Continue`). This is important for real-time and embedded applications.

### Backward Integration

The integrator supports backward integration ($t_f < t_0$) by passing a negative initial step size. The step-size controller computes `h_next` as a positive magnitude; the integration loop applies the correct sign.

---

## 8. Numerical Considerations

### IEEE 754 Pitfalls

- **Catastrophic cancellation**: The error estimate $k_0 + k_{10} - k_{11} - k_{12}$ involves subtraction of similar-magnitude quantities. When the error is very small relative to the solution, this can lose significant digits. The tolerance scaling denominator $a_i + r_i |y_i|$ mitigates this by normalizing appropriately.

- **Overflow in $r^3$**: For two-body problems, computing $r^3 = (x^2 + y^2 + z^2)^{3/2}$ can overflow for large distances. Factoring as `r * r * r` is preferred over `r.powi(3)` for clarity, though both are equivalent for positive values.

- **Tolerance floor**: The absolute tolerance provides a floor below which the error is not scaled. Without `atol`, components near zero would require $\epsilon \to 0$, which is unachievable in finite precision.

### Tolerance Selection Guidelines

| Application | `atol` | `rtol` | Notes |
|-------------|--------|--------|-------|
| High-precision orbit determination | $10^{-12}$ | $10^{-12}$ | Sub-mm position accuracy |
| General trajectory propagation | $10^{-10}$ | $10^{-10}$ | Standard engineering |
| Quick survey / screening | $10^{-6}$ | $10^{-6}$ | Fast, approximate |
| Mixed position/velocity (km, km/s) | `atol=[1e-12, 1e-12, 1e-12, 1e-15, 1e-15, 1e-15]` | $10^{-12}$ | Scale-aware |

**Validation criterion**: At `tol = 1e-12`, energy drift for a Keplerian orbit should be $< 10^{-10}$ over one orbital period.

### When RKF78 May Struggle

- **Stiff problems**: RKF78 is an explicit method and will take extremely small steps on stiff systems. Use an implicit method (e.g., Radau IIA) instead.
- **Discontinuous right-hand sides**: Thrust on/off, atmospheric drag with sharp density models. Use event detection to stop at discontinuities and restart.
- **Very long integrations**: Error accumulates over many steps. Consider compensated (Kahan) summation for the state update, or use a symplectic method if the problem has Hamiltonian structure.

---

## 9. References

1. Fehlberg, E. (1968). *"Classical Fifth-, Sixth-, Seventh-, and Eighth-Order Runge-Kutta Formulas with Stepsize Control"*. NASA Technical Report R-287. [https://ntrs.nasa.gov/citations/19680027281](https://ntrs.nasa.gov/citations/19680027281)

2. Hairer, E., Nørsett, S.P., & Wanner, G. (1993). *"Solving Ordinary Differential Equations I: Nonstiff Problems"*. Springer Series in Computational Mathematics, Vol. 8.

3. Hairer, E., Lubich, C., & Wanner, G. (2006). *"Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations"*. Springer.

4. Brent, R.P. (1973). *"Algorithms for Minimization without Derivatives"*. Prentice-Hall.

5. Montenbruck, O. & Gill, E. (2000). *"Satellite Orbits: Models, Methods and Applications"*. Springer.

6. Vallado, D.A. (2013). *"Fundamentals of Astrodynamics and Applications"*, 4th Edition. Microcosm Press.
