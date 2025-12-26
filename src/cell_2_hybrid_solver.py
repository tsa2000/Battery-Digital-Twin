# ==========================================
# CELL 2: Hybrid PINN + Classical Physics Solver
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import torch

def predict_hybrid_with_full_uncertainty(model, t_max_seconds, v_kmh=0.0, n_samples=100):
    """
    Two-stage hybrid solver:
    Stage 1 (0-60s): PINN handles complex thermal runaway dynamics
    Stage 2 (60s+): Classical heat equation for efficient long-term cooling

    Uncertainty propagation via Monte Carlo sampling throughout simulation.
    """
    t_pinn_limit = 60.0
    t_pinn = torch.linspace(0, min(t_max_seconds, t_pinn_limit), 200).view(-1, 1)
    v_input = torch.full_like(t_pinn, v_kmh)

    # Stage 1: PINN with Monte Carlo dropout
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(t_pinn, v_input).numpy().flatten())
    preds = np.array(preds)

    T_pinn_mean = preds.mean(axis=0)
    T_pinn_std = preds.std(axis=0)

    if t_max_seconds <= t_pinn_limit:
        return t_pinn.flatten().numpy(), T_pinn_mean, T_pinn_std

    # Stage 2: Classical physics with parameter uncertainty
    dt = 1.0
    current_t = t_pinn_limit
    T_samples = preds[:, -1]  # Initial distribution from PINN

    t_physics = []
    T_physics_mean = []
    T_physics_std = []

    v_ms = abs(v_kmh) / 3.6
    h_nominal = H_BASELINE + 4.0 * (v_ms ** 0.8)
    h_base = max(H_BASELINE, h_nominal)

    while current_t < t_max_seconds:
        current_t += dt

        # Propagate uncertainty through physics equations
        new_samples = []
        for T_sample in T_samples:
            # Model parameter uncertainty (from experimental variability)
            h_val = h_base * np.random.normal(1.0, 0.05)  # ±5% on h
            eps_val = EPSILON * np.random.normal(1.0, 0.02)  # ±2% on epsilon

            # Heat transfer equations
            Q_conv = h_val * A * (T_sample - T_INF)
            Q_rad = eps_val * SIGMA * A * (
                (T_sample + 273.15)**4 - (T_INF + 273.15)**4
            )

            dTdt = -(Q_conv + Q_rad) / (M * CP)
            new_samples.append(T_sample + dTdt * dt)

        T_samples = np.array(new_samples)
        t_physics.append(current_t)
        T_physics_mean.append(T_samples.mean())
        T_physics_std.append(T_samples.std())

    # Combine stages
    all_t = np.concatenate([t_pinn.flatten().numpy(), np.array(t_physics)])
    all_T = np.concatenate([T_pinn_mean, np.array(T_physics_mean)])
    all_std = np.concatenate([T_pinn_std, np.array(T_physics_std)])

    return all_t, all_T, all_std

# Run 3-hour simulation
print("Running long-term simulation...")

total_hours = 3
t_sim, T_sim, std_sim = predict_hybrid_with_full_uncertainty(
    model,
    t_max_seconds=3600*total_hours,
    v_kmh=0.0,
    n_samples=100
)

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[3, 1])

# Main temperature plot
upper = T_sim + (std_sim * 2)
lower = T_sim - (std_sim * 2)

ax1.fill_between(t_sim/3600, lower, upper,
                 color='red', alpha=0.25,
                 label='95% Confidence Interval')
ax1.plot(t_sim/3600, T_sim, 'r-', linewidth=3,
         label='Mean Temperature')
ax1.axhline(T_INF, color='cyan', linestyle='--', linewidth=2,
            label=f'Ambient ({T_INF}°C)')
ax1.axvline(60/3600, color='yellow', linestyle=':', linewidth=2,
            label='Stage Transition (60s)', alpha=0.7)

ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title(f'Long-Term Simulation ({total_hours} Hours)\n'
              f'Final: {T_sim[-1]:.2f}°C ± {std_sim[-1]*2:.4f}°C',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Uncertainty evolution plot
ax2.fill_between(t_sim/3600, 0, std_sim*2, color='orange', alpha=0.4)
ax2.plot(t_sim/3600, std_sim*2, 'orange', linewidth=2)
ax2.axvline(60/3600, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.set_xlabel('Time (Hours)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Uncertainty (±°C)', fontsize=11)
ax2.set_title('Uncertainty Evolution (95% CI)', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistics
print(f"\nSimulation Results:")
print(f"{'='*60}")
print(f"  Peak Temperature: {max(T_sim):.2f}°C")
print(f"  Final Temperature: {T_sim[-1]:.2f}°C")
print(f"  Total Cooling: {max(T_sim) - T_sim[-1]:.2f}°C")
print(f"\nUncertainty Analysis:")
print(f"  Peak Uncertainty (PINN): ±{max(std_sim[:200])*2:.2f}°C")
print(f"  At 10 minutes: ±{std_sim[np.argmin(np.abs(t_sim-600))]*2:.3f}°C")
print(f"  At 1 hour: ±{std_sim[np.argmin(np.abs(t_sim-3600))]*2:.3f}°C")
print(f"  At 3 hours: ±{std_sim[-1]*2:.4f}°C")
print(f"{'='*60}")
