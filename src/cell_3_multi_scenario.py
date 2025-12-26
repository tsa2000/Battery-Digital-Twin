# ==========================================
# CELL 3: Multi-Scenario Analysis
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
import torch

def solve_scenario_with_uncertainty(model, duration_minutes, v_kmh,
                                     ambient_temp_c, n_samples=100):
    """
    Scenario-specific simulation with ambient temperature adjustment.
    """
    t_max = duration_minutes * 60
    t_pinn_limit = 60.0

    # PINN stage
    t_pinn = torch.linspace(0, min(t_max, t_pinn_limit), 200).view(-1, 1)
    v_input = torch.full_like(t_pinn, v_kmh)

    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(t_pinn, v_input).numpy().flatten())
    preds = np.array(preds)

    # Apply ambient temperature offset
    delta_T = ambient_temp_c - 23.0
    T_pinn_mean = preds.mean(axis=0) + delta_T
    T_pinn_std = preds.std(axis=0)

    if t_max <= t_pinn_limit:
        return t_pinn.flatten().numpy(), T_pinn_mean, T_pinn_std

    # Classical physics stage
    dt = 1.0
    current_t = t_pinn_limit
    T_samples = preds[:, -1] + delta_T

    t_physics, T_physics_mean, T_physics_std = [], [], []

    v_ms = abs(v_kmh) / 3.6
    h_nominal = H_BASELINE + 4.0 * (v_ms ** 0.8)
    h_base = max(H_BASELINE, h_nominal)

    while current_t < t_max:
        current_t += dt

        new_samples = []
        for T_sample in T_samples:
            h_val = h_base * np.random.normal(1.0, 0.05)
            eps_val = EPSILON * np.random.normal(1.0, 0.02)

            Q_conv = h_val * A * (T_sample - ambient_temp_c)
            Q_rad = eps_val * SIGMA * A * (
                (T_sample + 273.15)**4 - (ambient_temp_c + 273.15)**4
            )
            dT = -(Q_conv + Q_rad) / (M * CP)
            new_samples.append(T_sample + dT * dt)

        T_samples = np.array(new_samples)
        t_physics.append(current_t)
        T_physics_mean.append(T_samples.mean())
        T_physics_std.append(T_samples.std())

    return (
        np.concatenate([t_pinn.flatten().numpy(), t_physics]),
        np.concatenate([T_pinn_mean, T_physics_mean]),
        np.concatenate([T_pinn_std, T_physics_std])
    )

# Define test scenarios
scenarios = [
    {'name': 'City Cruising', 'duration': 10, 'velocity': 60.0,
     'ambient': 23.0, 'color': 'blue', 'linestyle': '-'},
    {'name': 'Desert Parking', 'duration': 20, 'velocity': 0.0,
     'ambient': 45.0, 'color': 'red', 'linestyle': '--'},
    {'name': 'Winter Drive', 'duration': 15, 'velocity': 120.0,
     'ambient': -10.0, 'color': 'cyan', 'linestyle': '-.'}
]

print("Running multi-scenario analysis...")
print("="*60)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for scenario in scenarios:
    print(f"  Simulating: {scenario['name']}...")
    t, T, std = solve_scenario_with_uncertainty(
        model,
        scenario['duration'],
        scenario['velocity'],
        scenario['ambient']
    )

    # Temperature plot
    ax1.fill_between(t/60, T - std*2, T + std*2,
                     color=scenario['color'], alpha=0.15)
    ax1.plot(t/60, T,
             color=scenario['color'],
             linewidth=2.5,
             linestyle=scenario['linestyle'],
             label=f"{scenario['name']} ({scenario['velocity']} km/h, {scenario['ambient']}°C)")

    # Uncertainty plot
    ax2.plot(t/60, std*2,
             color=scenario['color'],
             linewidth=2,
             linestyle=scenario['linestyle'],
             label=scenario['name'])

    print(f"      Peak: {max(T):.1f}°C | Final: {T[-1]:.1f}°C | Max Unc: ±{max(std)*2:.2f}°C")

# Format plots
ax1.axvline(1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5,
            label='Stage Transition')
ax1.set_xlabel('Time (Minutes)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Multi-Scenario Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

ax2.axvline(1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax2.set_xlabel('Time (Minutes)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Uncertainty (±°C, 95% CI)', fontsize=12, fontweight='bold')
ax2.set_title('Uncertainty Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*60)
print("Analysis complete.")
