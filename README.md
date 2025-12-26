
# 🔋 0D Battery Thermal Runaway Digital Twin

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Physics-Informed Neural Network (PINN) for real-time thermal runaway prediction and safety assessment of 18650 Li-ion batteries.

---

## 🌟 Features

- **🧠 Hybrid PINN Solver**: Combines physics-informed ML with classical heat transfer
- **📊 Uncertainty Quantification**: Monte Carlo sampling (100 iterations) with 95% confidence intervals
- **💬 Natural Language Interface**: AI agent understands plain English queries
- **📈 Comprehensive Visualization**: 4 synchronized plots tracking temperature, uncertainty, cooling rate, and energy
- **⚠️ Safety Assessment**: Real-time risk evaluation with actionable recommendations
- **🎯 Multi-Scenario Support**: From desert parking to highway driving

---

## 🚀 Quick Start

### Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsa2000/Battery-Digital-Twin/blob/main/Battery_Digital_Twin_PINN.ipynb)

### Run Locally

Clone the repository
git clone https://github.com/tsa2000/Battery-Digital-Twin.git
cd Battery-Digital-Twin

Install dependencies
pip install torch numpy matplotlib gradio

Open the notebook
jupyter notebook Battery_Digital_Twin_PINN.ipynb

text

---

## 📊 Example Queries

The interface accepts natural language queries:

- `"What happens if the vehicle cruises at 100 km/h for 10 minutes?"`
- `"Simulate desert parking at 45°C for 20 minutes"`
- `"Winter driving at 120 km/h in -10°C"`
- `"Highway scenario at 110 km/h in summer (35°C)"`
- `"City driving at 60 km/h for 15 minutes"`

---

## 🔬 Model Architecture

### Zero-Dimensional (0D) Lumped Parameter Approach

**Why 0D?**
- ✅ Real-time performance
- ✅ Sufficient accuracy for safety assessment
- ✅ 100× faster than CFD
- ✅ Ideal for digital twin applications

### Hybrid Solver

**Stage 1 (0-60s): PINN**
- Handles complex thermal runaway dynamics
- Physics-informed loss function
- Trained on 12,000 epochs

**Stage 2 (60s+): Classical Physics**
- Efficient long-term cooling simulation
- Energy balance: `m·Cp·dT/dt = -Q_conv - Q_rad`

### Energy Balance Equation

m·Cp·dT/dt = Q_gen(t) - h·A·(T - T_amb) - ε·σ·A·(T⁴ - T_amb⁴)

text

**Where:**
- `h = 5 + 4·v^0.8` [W/m²·K] (velocity-dependent convection)
- `ε = 0.85` (emissivity)
- `σ = 5.67×10⁻⁸` W/m²·K⁴ (Stefan-Boltzmann constant)

---

## 📐 Cell Specifications

| Parameter | Value | Unit |
|-----------|-------|------|
| **Model** | 18650 LG MJ1 | - |
| **Chemistry** | NMC | - |
| **Mass** | 42 | g |
| **Specific Heat** | 800 | J/kg·K |
| **Surface Area** | 0.004185 | m² |
| **Emissivity** | 0.85 | - |
| **Total Energy Released** | ~32.2 | kJ |

---

## 🎯 Key Results

### Counter-intuitive Finding

**High-speed driving in extreme heat is safer than stationary parking in moderate heat!**

| Scenario | Speed | Ambient | Duration | Peak | Final | Status |
|----------|-------|---------|----------|------|-------|--------|
| **High-speed Extreme Heat** | 120 km/h | 55°C | 15 min | 970.7°C | 55.1°C | ✅ **SAFE** |
| **Desert Parking** | 0 km/h | 45°C | 20 min | 982.3°C | 84.6°C | ⚠️ **CAUTION** |

**Insight**: Airflow velocity (convection coefficient `h`) dominates over ambient temperature difference for post-thermal runaway cooling.

**Design Implication**: Active cooling systems are more critical than passive thermal insulation in hot climates.

---

## 📈 Output Examples

The interface provides:

1. **Temperature Evolution** - with peak identification and 95% CI bands
2. **Uncertainty Quantification** - showing confidence over time
3. **Instantaneous Cooling Rate** - thermal dynamics visualization
4. **Cumulative Energy Dissipation** - heat removal tracking

Each output includes:
- ✅ Safety status (SAFE / CAUTION / ALERT)
- ✅ Time-to-safe estimation
- ✅ Risk-specific recommendations
- ✅ Comparison with reference scenarios

---

## 🛠️ Technical Details

### Training

- **Epochs**: 12,000
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: Physics-informed (PDE residual + boundary conditions)
- **Regularization**: Dropout (5% per layer)

### Uncertainty Sources

- **Model Uncertainty**: Dropout-based (Monte Carlo, n=100)
- **Parameter Uncertainty**: h (±5%), ε (±2%)

### Performance

- **Training Time**: ~15 minutes (GPU)
- **Inference Time**: <1 second per scenario
- **Confidence Intervals**: 95% CI

---

## 📚 Scientific Basis

This work is based on experimental research by:

**Coman, P. T., Mátéfi-Tempfli, S., Veje, C. T., & White, R. E. (2022)**  
*"Simplified Thermal Runaway Model for Lithium-Ion Cells"*  
**Journal of The Electrochemical Society**, 169(4), 040516.  
[https://doi.org/10.1149/1945-7111/ac62c6](https://doi.org/10.1149/1945-7111/ac62c6)

---

## 🔮 Future Work

- ⏰ **Time-varying conditions**: Emergency stop scenarios (crash safety)
- 🔋 **Multi-cell modeling**: Battery pack thermal propagation
- 🌡️ **Spatial gradients**: Extension to 1D/2D models
- 🔌 **BMS integration**: Real-time monitoring and control
- 🧪 **Experimental validation**: Lab testing with different cell chemistries

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests
- ⭐ Star the repository if you find it useful!

---

## 👤 Author
**THAER ABUSHAWER**
**Mechanical Engineering Graduate Student**  
Focus: Thermal Systems & Battery Management  
Research: Physics-Informed Machine Learning for Energy Applications

---

## 🙏 Acknowledgments

- Dr. Paul T. Coman (University of South Carolina) for research guidance
- Experimental data from Coman et al. (2022) publication
- Open-source community for PyTorch and Gradio frameworks

---

## 📧 Contact

For questions, collaborations, or feedback:
- 📫 Open an issue on GitHub
- thaer199@gmail.com

---

## ⭐ Citation

If you use this work in your research, please cite:

@software{battery_digital_twin_2025,
author = {Thaer Abushawer},
title = {0D Battery Thermal Runaway Digital Twin},
year = {2025},
url = {https://github.com/tsa2000/Battery-Digital-Twin}
}

text

---

**Made with ❤️ and ⚡ for safer battery systems** 
