# 🏆 World Cup 2026 Prediction Engine

An advanced machine learning system to predict international soccer matches and simulate the 2026 World Cup tournament using Monte Carlo methods and ensemble learning.

---

## 📊 Overview

This project combines team statistics, historical match data, and sophisticated ML models to forecast World Cup outcomes. The engine simulates thousands of tournament iterations to estimate championship probabilities and expected match results.

**Key Capabilities:**
- Match outcome prediction (Win/Draw/Loss)
- Expected Goals (xG) forecasting
- Full tournament simulation with 10,000+ iterations
- Championship probability estimation

---

## 🔧 Methodology

### Data Pipeline
- **Data Source**: Kaggle International Football Statistics
- **Features**: Elo ratings, team overall ratings, attack/defense stats, match neutrality
- **Preprocessing**: Label encoding, feature normalization, null value handling

### Machine Learning Models

| Component | Model | Purpose | Metric |
|-----------|-------|---------|--------|
| **Match Outcomes** | XGBoost Classifier | Predict Win/Draw/Loss | 62% Accuracy |
| **Expected Goals** | Gradient Boosting Regressor | Forecast goals scored | 1.11 MAE |

### Architecture
- **Object-Oriented Design**: `Team`, `WorldCupSimulator` classes
- **Ensemble Approach**: Combines multiple signals for robust predictions
- **Monte Carlo Simulation**: Accounts for tournament randomness and edge cases

---

## 🎯 Key Results

### 2026 Championship Predictions
| Rank | Team | Probability | Elo | Overall |
|------|------|-------------|-----|---------|
| 🥇 | Argentina | 67.9% | 2115 | 86 |
| 🥈 | France | 18.5% | 2080 | 85 |
| 🥉 | Brazil | 8.2% | 2050 | 84 |
| 4️⃣ | England | 3.1% | 2010 | 83 |
| 5️⃣ | Spain | 1.8% | 1980 | 83 |

### Model Performance
- **Classification Accuracy**: 62%
- **xG Prediction MAE**: 1.11 goals
- **Simulation Stability**: Consistent over 5,000+ iterations

---

## 📁 Project Structure

```
world-cup-predictor/
├── notebooks/
│   └── World_Cup_Prediction.ipynb    # Main analysis & simulation
├── data/
│   └── teams_match_features.csv      # Historical match data
├── output/
│   └── world_cup_2026_projections.csv # Championship probabilities
├── README.md                          # This file
└── requirements.txt                   # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/joashpark/world-cup-predictor.git
cd world-cup-predictor

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebook

```bash
jupyter notebook notebooks/World_Cup_Prediction.ipynb
```

The notebook executes in stages:
1. **Data Loading & Cleaning** - Prepares historical match data
2. **Model Training** - Trains XGBoost and Gradient Boosting models
3. **Tournament Simulation** - Runs 5,000+ tournament iterations
4. **Visualization & Export** - Generates charts and CSV projections

---

## 📈 Visualizations

The project generates:
- **Championship Probability Bar Chart** - Win % for top contenders
- **Match Outcome Heatmaps** - Historical accuracy by team
- **Expected Goals Scatter Plot** - Predicted vs. actual goals

---

## 🔍 How It Works

### Match Prediction Pipeline

```
Team Statistics (Elo, Overall, Attack, Defense)
    ↓
Feature Engineering (Differences, Ratios)
    ↓
XGBoost Classification Model
    ↓
Outcome Probabilities (Win/Draw/Loss)
    ↓
Tournament Simulation (Bracket Progression)
```

### Example: Argentina vs France

```python
t1 = Team('Argentina', 2100, 85, 87, 83)
t2 = Team('France', 2050, 84, 88, 82)
result = simulator.simulate_match(t1, t2)  # Output: "Argentina"
```

The model considers:
- Elo differential: Argentina +50 (historical strength)
- Overall rating differential: +1 (marginal)
- Attack vs Defense matchups
- Home/neutral field advantage

---

## 📊 Data Features

| Feature | Description | Source |
|---------|-------------|--------|
| `home_elo` | Home team Elo rating | World Football Elo |
| `away_elo` | Away team Elo rating | World Football Elo |
| `overall_diff` | Overall team rating difference | Kaggle FIFA Stats |
| `attack_diff` | Attack stat differential | Team Stats |
| `defense_diff` | Defense stat differential | Team Stats |
| `is_neutral` | Neutral venue (1 = yes, 0 = no) | Match Info |
| `home_goals` | Goals scored by home team | Match Record |
| `away_goals` | Goals scored by away team | Match Record |

---

## 🎓 Model Insights

### XGBoost Classifier
- **Training Samples**: 5,000+ historical matches
- **Feature Importance**: Elo differential is the strongest predictor
- **Class Distribution**: Win (45%), Draw (25%), Loss (30%)
- **Regularization**: `use_label_encoder=False`, `eval_metric='mlogloss'`

### Gradient Boosting Regressor (xG)
- **n_estimators**: 100 trees
- **Learning Rate**: 0.1
- **Max Depth**: 3
- **Predicts**: Expected goals for home and away teams

---

## 📌 Limitations & Considerations

- **Historical Bias**: Model trained on past data; doesn't account for roster changes
- **Injuries & Form**: No real-time injury or current form adjustments
- **Tournament Format**: Simplified knockout bracket (actual World Cup has group stages)
- **Accuracy Ceiling**: 62% reflects inherent unpredictability of sports
- **Data Recency**: Trained on historical data; updates needed seasonally

---

## 🔮 Future Enhancements

- [ ] Integrate real-time injury databases
- [ ] Add player-level performance metrics
- [ ] Implement group stage simulation
- [ ] Include betting line calibration
- [ ] Create API endpoint for live predictions
- [ ] Add confidence intervals to projections
- [ ] Retrain models with 2024-2025 match data

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or suggest improvements
- Submit pull requests with enhancements
- Share alternative modeling approaches

---

## 📄 License

This project is open source. Feel free to use and modify for educational purposes.

---

## 📞 Contact & Resources

- **Author**: joashpark
- **Repository**: [world-cup-predictor](https://github.com/joashpark/world-cup-predictor)
- **Data Source**: [Kaggle International Football Statistics](https://www.kaggle.com/)
- **Inspiration**: World Football Elo Ratings

---

## 🎯 Disclaimer

*This project is for educational and entertainment purposes. Predictions are based on historical data and machine learning models. Actual tournament outcomes will vary. Use results responsibly and never make betting decisions based solely on these predictions.*

---

**Last Updated**: July 2026 | **Model Version**: 1.0
