# ==========================================
# CELL 1: PINN Training for Thermal Runaway
# ==========================================
import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Physical constants (Coman et al. 2022, J. Electrochem. Soc.)
M = 0.042      # Cell mass [kg]
CP = 800.0     # Specific heat [J/kg/K]
A = 0.004185   # Surface area [m^2]
T_INF = 23.0   # Ambient temperature [C]
H_BASELINE = 5.0  # Natural convection coefficient [W/m^2/K]
SIGMA = 5.67e-8   # Stefan-Boltzmann constant [W/m^2/K^4]
EPSILON = 0.85    # Surface emissivity [-]

# Normalization scales
TIME_SCALE = 60.0
T_SCALE = 1000.0
V_SCALE = 200.0

def get_q_tr(t):
    """
    Heat generation rate during thermal runaway.
    Extracted from Figure 3 in reference paper.
    Triangular profile: 0-1s ramp up, 1-6s ramp down.
    Total energy: 32.2 kJ (conservative worst-case).
    """
    q_peak = 11040.0  # Peak power [W]
    val = torch.zeros_like(t)

    # Ramp up phase
    mask1 = (t < 1.0) & (t >= 0.0)
    val[mask1] = q_peak * t[mask1]

    # Ramp down phase
    mask2 = (t >= 1.0) & (t <= 6.0)
    val[mask2] = q_peak * (1.0 - (t[mask2] - 1.0) / 5.0)

    return val

def get_h(v_kmh):
    """
    Forced convection coefficient as function of velocity.
    Empirical correlation: h = h_0 + C * v^0.8
    """
    v_ms = torch.abs(v_kmh) / 3.6
    h_forced = H_BASELINE + 4.0 * (v_ms ** 0.8)
    return torch.clamp(h_forced, min=H_BASELINE)

class BatteryPINN_Uncertainty(nn.Module):
    """
    Physics-Informed Neural Network with dropout for uncertainty quantification.
    Architecture: 2 inputs (time, velocity) -> 3 hidden layers -> 1 output (temperature)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Dropout(0.05),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, t, v):
        """Forward pass with input normalization."""
        inputs = torch.cat([t/TIME_SCALE, v/V_SCALE], dim=1)
        return self.net(inputs) * T_SCALE

def physics_loss(model, t, v):
    """
    Compute physics-informed loss based on energy balance:
    m * Cp * dT/dt = Q_gen - Q_conv - Q_rad
    """
    t.requires_grad = True
    T_pred = model(t, v)

    # Automatic differentiation for dT/dt
    dTdt = torch.autograd.grad(T_pred, t,
                                grad_outputs=torch.ones_like(T_pred),
                                create_graph=True)[0]

    # Heat generation and losses
    Q_gen = get_q_tr(t)
    h_val = get_h(v)
    Q_conv = h_val * A * (T_pred - T_INF)
    Q_rad = EPSILON * SIGMA * A * ((T_pred+273.15)**4 - (T_INF+273.15)**4)

    # Energy balance residual
    residual = (M * CP * dTdt) - (Q_gen - Q_conv - Q_rad)
    return torch.mean((residual / (M*CP))**2)

# Initialize model and optimizer
model = BatteryPINN_Uncertainty()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=500
)

loss_weight_ic = 100.0
best_loss = float('inf')
best_model_state = None

print("Starting PINN training...")
print("="*60)

for epoch in range(12000):
    optimizer.zero_grad()

    # Sample collocation points - focus on event phase
    t_event = torch.rand(2000, 1) * 10.0
    v_event = torch.rand(2000, 1) * 200.0

    t_cool = 10.0 + torch.rand(1000, 1) * 50.0
    v_cool = torch.rand(1000, 1) * 200.0

    t_phy = torch.cat([t_event, t_cool], dim=0)
    v_phy = torch.cat([v_event, v_cool], dim=0)

    # Physics loss
    loss_phy = physics_loss(model, t_phy, v_phy)

    # Initial condition enforcement
    t_ic = torch.zeros(500, 1)
    v_ic = torch.rand(500, 1) * 200.0
    loss_ic = torch.mean((model(t_ic, v_ic) - T_INF)**2)

    # Total loss
    total_loss = loss_phy + loss_weight_ic * loss_ic
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)

    # Track best model
    if total_loss.item() < best_loss:
        best_loss = total_loss.item()
        best_model_state = copy.deepcopy(model.state_dict())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.6f} | Best: {best_loss:.6f}")

print("="*60)
print("Training complete.")

# Load best model
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), "battery_pinn_uncertainty.pth")
print(f"Model saved. Final loss: {best_loss:.6f}")

# Quick validation
def predict_with_uncertainty(model, t, v, n_samples=100):
    """Monte Carlo prediction using dropout."""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(t, v).numpy().flatten())
    preds = np.array(preds)
    return preds.mean(axis=0), preds.std(axis=0)

t_vals = torch.linspace(0, 60, 200).view(-1, 1)
v_vals = torch.zeros_like(t_vals)
mean, std = predict_with_uncertainty(model, t_vals, v_vals)

plt.figure(figsize=(11, 6))
plt.fill_between(t_vals.flatten(), mean - 2*std, mean + 2*std,
                 color='red', alpha=0.3, label='95% CI')
plt.plot(t_vals, mean, 'r-', linewidth=2.5, label='Mean Prediction')
plt.axhline(T_INF, color='cyan', linestyle='--', label='Ambient')
plt.title(f"Validation: Peak {mean.max():.1f}°C, Uncertainty ±{max(std)*2:.2f}°C",
          fontsize=13, fontweight='bold')
plt.xlabel("Time (s)", fontsize=11)
plt.ylabel("Temperature (°C)", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nValidation Results:")
print(f"  Peak Temperature: {mean.max():.1f}°C")
print(f"  Max Uncertainty: ±{max(std)*2:.2f}°C")
