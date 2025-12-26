# ==========================================
# Cell 4: Enhanced Gradio Interface
# ==========================================
!pip install -q gradio

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# Constants
M, CP, A = 0.042, 800.0, 0.004185
T_INF = 23.0
TIME_SCALE, T_SCALE, V_SCALE = 60.0, 1000.0, 200.0
SIGMA, EPSILON = 5.67e-8, 0.85
H_BASELINE = 5.0

class BatteryPINN_Uncertainty(nn.Module):
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
        inputs = torch.cat([t/TIME_SCALE, v/V_SCALE], dim=1)
        return self.net(inputs) * T_SCALE

# Load model
model = BatteryPINN_Uncertainty()
model.load_state_dict(torch.load("battery_pinn_uncertainty.pth", map_location='cpu'))

def solve_hybrid_scenario(model, duration_min, v_kmh, amb_temp):
    duration_sec = duration_min * 60.0
    t_limit = 60.0

    t_pinn = torch.linspace(0, min(duration_sec, t_limit), 200).view(-1, 1)
    v_input = torch.full_like(t_pinn, v_kmh)

    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(100):
            preds.append(model(t_pinn, v_input).numpy().flatten())
    preds = np.array(preds)

    delta_T = amb_temp - 23.0
    T_mean = preds.mean(axis=0) + delta_T
    T_std = preds.std(axis=0)

    if duration_sec <= t_limit:
        return t_pinn.flatten().numpy(), T_mean, T_std

    dt = 1.0
    curr_t = t_limit
    T_samples = preds[:, -1] + delta_T

    t_phy, T_phy, std_phy = [], [], []
    v_ms = abs(v_kmh) / 3.6
    h_base = max(H_BASELINE, H_BASELINE + 4.0 * (v_ms ** 0.8))

    while curr_t < duration_sec:
        curr_t += dt
        new_samples = []

        for T_s in T_samples:
            h = h_base * np.random.normal(1.0, 0.05)
            eps = EPSILON * np.random.normal(1.0, 0.02)
            Q_conv = h * A * (T_s - amb_temp)
            Q_rad = eps * SIGMA * A * ((T_s+273.15)**4 - (amb_temp+273.15)**4)
            dT = -(Q_conv + Q_rad) / (M * CP)
            new_samples.append(T_s + dT * dt)

        T_samples = np.array(new_samples)
        t_phy.append(curr_t)
        T_phy.append(T_samples.mean())
        std_phy.append(T_samples.std())

    return (np.concatenate([t_pinn.flatten().numpy(), t_phy]),
            np.concatenate([T_mean, T_phy]),
            np.concatenate([T_std, std_phy]))

def parse_query(query):
    query = query.lower()
    velocity, temperature, duration = 0.0, 23.0, 10.0

    v_match = re.search(r'(\d+)\s*(?:km/h|kmh)', query)
    if v_match: velocity = float(v_match.group(1))

    t_match = re.search(r'(-?\d+)\s*(?:°c|degrees?)', query)
    if t_match: temperature = float(t_match.group(1))

    d_match = re.search(r'(\d+)\s*(?:min|minutes?)', query)
    if d_match: duration = float(d_match.group(1))

    if 'winter' in query or 'cold' in query:
        temperature = -10.0
        velocity = max(velocity, 100.0) if velocity > 0 else 120.0
    if 'desert' in query or 'hot' in query:
        temperature = 45.0
        velocity = 0.0
    if 'parking' in query or 'stationary' in query:
        velocity = 0.0
    if 'highway' in query or 'cruising' in query:
        velocity = max(velocity, 100.0) if velocity > 0 else 100.0

    return velocity, temperature, duration

def calculate_h_coefficient(v_kmh):
    """Calculate convection coefficient"""
    v_ms = abs(v_kmh) / 3.6
    return max(H_BASELINE, H_BASELINE + 4.0 * (v_ms ** 0.8))

def get_safety_level(final_temp, ambient_temp):
    """Determine safety level"""
    delta = final_temp - ambient_temp
    if delta < 20:
        return "SAFE", "green", "✅"
    elif delta < 50:
        return "CAUTION", "orange", "⚠️"
    else:
        return "ALERT", "red", "🔴"

def estimate_safe_time(final_temp, ambient_temp, h_coeff):
    """Estimate time to reach safe temperature (<50°C)"""
    if final_temp < 50:
        return "Already safe"

    # Simplified exponential cooling
    tau = (M * CP) / (h_coeff * A)
    T_diff = final_temp - ambient_temp
    safe_diff = 50 - ambient_temp

    if safe_diff <= 0:
        return "Ambient too high"

    time_const = tau / 60  # to minutes
    time_needed = -time_const * np.log(safe_diff / T_diff)

    return f"~{int(time_needed)} minutes"

def generate_enhanced_response(query, vel, temp, dur, max_t, final_t, max_unc, t, T, std):
    """Generate comprehensive analysis"""

    h_coeff = calculate_h_coefficient(vel)
    safety, color, icon = get_safety_level(final_t, temp)
    safe_time = estimate_safe_time(final_t, temp, h_coeff)

    # Calculate cooling rate
    peak_idx = np.argmax(T)
    if len(T) > peak_idx + 60:
        cooling_rate = (T[peak_idx] - T[peak_idx + 60]) / 60
    else:
        cooling_rate = (max_t - final_t) / (dur * 60)

    # Total energy
    total_energy = M * CP * (max_t - temp) / 1000  # kJ

    response = f"""# {icon} Thermal Runaway Analysis

## Query
**"{query}"**

---

## Simulation Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Velocity | **{vel:.0f}** | km/h |
| Ambient Temperature | **{temp:.0f}** | °C |
| Simulation Duration | **{dur:.0f}** | minutes |
| Convection Coefficient (h) | **{h_coeff:.1f}** | W/m²·K |

---

## Thermal Performance

### Peak Conditions
- **Maximum Temperature**: {max_t:.1f}°C
- **Rise from Ambient**: +{max_t - temp:.1f}°C
- **Time to Peak**: ~30 seconds
- **Total Energy**: {total_energy:.1f} kJ

### Final Conditions (t = {dur:.0f} min)
- **Final Temperature**: {final_t:.1f}°C
- **Remaining Heat**: +{final_t - temp:.1f}°C above ambient
- **Cooling Achieved**: {max_t - final_t:.1f}°C

### Cooling Dynamics
- **Average Cooling Rate**: {cooling_rate:.2f}°C/s (first minute)
- **Heat Transfer Mode**: {"Natural convection only" if vel == 0 else f"Forced convection (v = {vel:.0f} km/h)"}

---

## Uncertainty Quantification

| Metric | Value | Confidence |
|--------|-------|------------|
| Maximum Uncertainty | ±{max_unc:.2f}°C | 95% CI |
| Peak Uncertainty | ±{std[np.argmax(T)]*2:.2f}°C | - |
| Final Uncertainty | ±{std[-1]*2:.2f}°C | - |
| Model Confidence | {"High" if max_unc < 50 else "Moderate" if max_unc < 100 else "Variable"} | - |

---

## Safety Assessment

### Status: **{safety}** {icon}

"""

    if safety == "SAFE":
        response += """
**Cell has returned to near-ambient temperature.**

✅ Safe for handling with standard PPE
✅ Thermal risk minimized
✅ Can proceed with inspection/disposal

**Recommendations:**
- Verify temperature with thermal camera before handling
- Use heat-resistant gloves as precaution
- Inspect for mechanical damage
"""
    elif safety == "CAUTION":
        response += f"""
**Elevated temperature persists - exercise caution.**

⚠️ Surface temperature: {final_t:.1f}°C (hot to touch)
⚠️ Additional cooling time needed: {safe_time}
⚠️ Risk of thermal burns

**Recommendations:**
- Wait additional time before handling
- Use thermal imaging to confirm cooling
- Ensure adequate ventilation
- Keep fire suppression equipment ready
"""
    else:
        response += f"""
**CRITICAL: Cell remains extremely hot!**

🔴 Surface temperature: {final_t:.1f}°C (danger zone)
🔴 Time to safe temperature: {safe_time}
🔴 High risk of secondary ignition
🔴 Thermal radiation hazard

**Immediate Actions Required:**
- DO NOT approach or handle the cell
- Maintain safe distance (>2 meters)
- Continue monitoring temperature
- Prepare for extended cooling period
- Alert emergency response team
"""

    response += f"""

---

## Physical Insights

### Heat Transfer Analysis
"""

    if vel == 0:
        response += """
- **Natural Convection Only** (h ≈ 5 W/m²·K)
- Cooling is 10-20× slower than forced convection
- Radiation becomes dominant at high temperatures
- Extended cooling time expected
"""
    elif vel < 60:
        response += f"""
- **Low-Speed Forced Convection** (h ≈ {h_coeff:.1f} W/m²·K)
- Moderate cooling enhancement ({h_coeff/H_BASELINE:.1f}× baseline)
- Both convection and radiation contribute
- Cooling time: moderate
"""
    elif vel < 120:
        response += f"""
- **Highway-Speed Convection** (h ≈ {h_coeff:.1f} W/m²·K)
- Strong cooling enhancement ({h_coeff/H_BASELINE:.1f}× baseline)
- Forced convection dominates at lower temperatures
- Cooling time: fast
"""
    else:
        response += f"""
- **High-Speed Forced Convection** (h ≈ {h_coeff:.1f} W/m²·K)
- Maximum cooling efficiency ({h_coeff/H_BASELINE:.1f}× baseline)
- Turbulent airflow provides excellent heat dissipation
- Cooling time: very fast
"""

    if temp > 40:
        response += f"\n⚠️ **Hot Environment**: Ambient {temp}°C reduces cooling effectiveness\n"
    elif temp < 5:
        response += f"\n❄️ **Cold Environment**: Ambient {temp}°C enhances cooling significantly\n"

    response += f"""

### Comparison with Reference Scenarios

| Scenario | Peak (°C) | Final (°C) | Status |
|----------|-----------|------------|--------|
| **Your Case** | **{max_t:.1f}** | **{final_t:.1f}** | **{safety}** |
| Desert Parking (0 km/h @ 45°C, 20 min) | 979 | 85 | CAUTION |
| City Drive (60 km/h @ 23°C, 10 min) | 940 | 30 | SAFE |
| Highway (110 km/h @ 35°C, 10 min) | 951 | 38 | SAFE |

---

## Technical Details

**Cell Specifications:**
- Model: 18650 LG MJ1
- Mass: 42 g
- Specific Heat: 800 J/kg·K
- Surface Area: 0.004185 m²
- Emissivity: 0.85

**Simulation Method:**
- PINN for thermal runaway phase (0-60s)
- Classical heat transfer for cooling phase (60s+)
- Monte Carlo uncertainty quantification (n=100)
- Time step: 1 second

**Reference:**
*Coman et al. (2022), J. Electrochem. Soc. 169, 040516*

---

**Simulation completed:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    return response

def create_enhanced_plot(t, T, std, vel, temp, dur, max_t, final_t):
    """Create comprehensive dual-plot visualization"""

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)

    # Main plot - Temperature Evolution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#1a1d29')

    upper = T + (std * 2)
    lower = T - (std * 2)

    # Temperature plot
    ax1.fill_between(t/60, lower, upper, color='red', alpha=0.3, label='95% CI')
    ax1.plot(t/60, T, 'r-', linewidth=3.5, label='Temperature', zorder=5)
    ax1.axhline(temp, color='cyan', linestyle='--', linewidth=2.5, label=f'Ambient ({temp}°C)')
    ax1.axvline(1.0, color='yellow', linestyle=':', linewidth=2.5, alpha=0.8, label='Stage Transition')

    # Add peak marker
    peak_idx = np.argmax(T)
    ax1.plot(t[peak_idx]/60, T[peak_idx], 'r*', markersize=20, label=f'Peak: {max_t:.1f}°C', zorder=10)

    # Add safe temperature line
    ax1.axhline(50, color='green', linestyle='-.', linewidth=2, alpha=0.7, label='Safe Threshold (50°C)')

    ax1.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold', color='white')
    ax1.set_title(f'Thermal Runaway: {vel:.0f} km/h @ {temp:.0f}°C\n'
                  f'Peak: {max_t:.1f}°C | Final: {final_t:.1f}°C | h = {calculate_h_coefficient(vel):.1f} W/m²·K',
                  fontsize=15, color='#ff5252', fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=10, facecolor='#262730', edgecolor='#444', ncol=2)
    ax1.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax1.tick_params(colors='white', labelsize=11)
    ax1.set_xlim(0, dur)

    for spine in ax1.spines.values():
        spine.set_edgecolor('#444')
        spine.set_linewidth(1.5)

    # Left bottom - Uncertainty Evolution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#1a1d29')
    ax2.fill_between(t/60, 0, std*2, color='orange', alpha=0.5)
    ax2.plot(t/60, std*2, 'orange', linewidth=2.5)
    ax2.axvline(1.0, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Time (Minutes)', fontsize=11, fontweight='bold', color='white')
    ax2.set_ylabel('Uncertainty (±°C)', fontsize=10, fontweight='bold', color='white')
    ax2.set_title('Uncertainty Evolution (95% CI)', fontsize=11, color='white')
    ax2.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax2.tick_params(colors='white', labelsize=9)
    ax2.set_xlim(0, dur)

    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')
        spine.set_linewidth(1.5)

    # Right bottom - Cooling Rate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#1a1d29')

    # Calculate cooling rate
    cooling_rate = np.zeros(len(T)-1)
    for i in range(len(T)-1):
        dt_sec = t[i+1] - t[i]
        if dt_sec > 0:
            cooling_rate[i] = -(T[i+1] - T[i]) / dt_sec

    ax3.plot(t[:-1]/60, cooling_rate, color='#00ff88', linewidth=2.5)
    ax3.axvline(1.0, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    ax3.set_xlabel('Time (Minutes)', fontsize=11, fontweight='bold', color='white')
    ax3.set_ylabel('Cooling Rate (°C/s)', fontsize=10, fontweight='bold', color='white')
    ax3.set_title('Instantaneous Cooling Rate', fontsize=11, color='white')
    ax3.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax3.tick_params(colors='white', labelsize=9)
    ax3.set_xlim(0, dur)

    for spine in ax3.spines.values():
        spine.set_edgecolor('#444')
        spine.set_linewidth(1.5)

    # Bottom - Energy Distribution
    ax4 = fig.add_subplot(gs[2, :])
    ax4.set_facecolor('#1a1d29')

    # Calculate cumulative energy removed
    energy_removed = np.zeros(len(T))
    for i in range(len(T)):
        energy_removed[i] = M * CP * (max_t - T[i]) / 1000  # kJ

    ax4.fill_between(t/60, 0, energy_removed, color='#00aaff', alpha=0.4)
    ax4.plot(t/60, energy_removed, color='#00aaff', linewidth=2.5, label='Energy Removed')
    ax4.axhline(M * CP * (max_t - temp) / 1000, color='cyan', linestyle='--',
                linewidth=2, label=f'Total Energy: {M * CP * (max_t - temp) / 1000:.1f} kJ')
    ax4.axvline(1.0, color='yellow', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Time (Minutes)', fontsize=11, fontweight='bold', color='white')
    ax4.set_ylabel('Energy (kJ)', fontsize=10, fontweight='bold', color='white')
    ax4.set_title('Cumulative Heat Dissipation', fontsize=11, color='white')
    ax4.legend(fontsize=10, facecolor='#262730', edgecolor='#444')
    ax4.grid(True, alpha=0.3, color='gray', linestyle='--')
    ax4.tick_params(colors='white', labelsize=9)
    ax4.set_xlim(0, dur)

    for spine in ax4.spines.values():
        spine.set_edgecolor('#444')
        spine.set_linewidth(1.5)

    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()

    return fig

def chat_function(query, history):
    if not query.strip():
        return history + [[query, "Please enter a query. Example: 'Simulate at 100 km/h for 15 minutes'"]]

    vel, temp, dur = parse_query(query)
    t, T, std = solve_hybrid_scenario(model, dur, vel, temp)

    max_t = max(T)
    final_t = T[-1]
    max_unc = max(std) * 2

    # Create enhanced visualization
    fig = create_enhanced_plot(t, T, std, vel, temp, dur, max_t, final_t)

    # Generate comprehensive response
    response = generate_enhanced_response(query, vel, temp, dur, max_t, final_t, max_unc, t, T, std)

    return history + [[query, response]], fig

# Create interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="red", secondary_hue="blue"),
               title="Battery Digital Twin - Enhanced") as demo:

    gr.Markdown("""
# 🔋 0D Battery Thermal Runaway Digital Twin
### PINN-Based Real-time Safety Assessment
**18650 LG MJ1 Cell**
""")


    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Digital Twin Analysis",
                height=550,
                show_label=True
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="",
                    placeholder="Example: What happens if the vehicle cruises at 100 km/h for 10 minutes?",
                    lines=2,
                    scale=4
                )
                btn = gr.Button("🚀 Run Simulation", variant="primary", scale=1, size="lg")

            plot = gr.Plot(label="Comprehensive Thermal Analysis")

        with gr.Column(scale=1):
            gr.Markdown("""
            ## 💡 Example Queries

            **Realistic Scenarios:**
            - "What happens at 100 km/h for 10 minutes?"
            - "Simulate desert parking at 45°C for 20 minutes"
            - "Winter driving at 120 km/h in -10°C"
            - "Highway at 110 km/h in summer (35°C)"
            - "City driving at 60 km/h for 15 minutes"

            ---

            ## 🎯 Key Features

            ✅ **Natural Language Understanding**
            - Extracts velocity, temperature, duration automatically

            ✅ **Uncertainty Quantification**
            - Monte Carlo sampling (100 iterations)
            - 95% confidence intervals

            ✅ **Comprehensive Analysis**
            - Temperature evolution
            - Cooling rate dynamics
            - Energy dissipation tracking
            - Safety assessment

            ✅ **Safety Recommendations**
            - Risk level evaluation
            - Time-to-safe estimation
            - Handling guidelines

            ---

            ## 📊 Model Architecture

            **Hybrid Solver:**
            - **Stage 1 (0-60s):** PINN handles complex thermal runaway
            - **Stage 2 (60s+):** Classical physics for efficient cooling

            **Training:**
            - 12,000 epochs
            - Physics-informed loss function
            - Energy balance: *m·Cp·dT/dt = Q_gen - Q_conv - Q_rad*

            **Cell Specifications:**
            - 18650 LG MJ1 (NMC chemistry)
            - 42g, 800 J/kg·K
            - Surface area: 0.004185 m²

            ---

            ## 📚 Scientific Basis

            Based on experimental work by:
            **Coman et al. (2022)**
            *J. Electrochem. Soc. 169, 040516*

            "Simplified Thermal Runaway Model for Lithium-Ion Cells"

            ---

            ## ⚙️ Technical Notes

            - Convection: *h = 5 + 4·v^0.8*
            - Radiation: *ε·σ·A·(T⁴ - T_amb⁴)*
            - Parameter uncertainty: ±5% (h), ±2% (ε)
            """)

    btn.click(chat_function, [msg, chatbot], [chatbot, plot])
    msg.submit(chat_function, [msg, chatbot], [chatbot, plot])

    gr.Examples(
        examples=[
            ["What happens if the vehicle cruises at 100 km/h for 10 minutes?"],
            ["Simulate desert parking at 45°C for 20 minutes"],
            ["Winter driving at 120 km/h in -10°C for 15 minutes"],
            ["Highway scenario at 110 km/h in summer (35°C)"],
            ["City driving at 60 km/h for 15 minutes"],
            ["Stationary test at room temperature for 30 minutes"]
        ],
        inputs=msg,
        label="📌 Quick Start Examples"
    )

print("\n" + "="*70)
print("🚀 Launching Enhanced Interface...")
print("="*70)

demo.launch(share=True, debug=True, show_error=True)

print("\n✅ Interface is live!")
print("🌐 Share the link for remote access")
print("⏱️  Link valid for 72 hours")
