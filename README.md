# VidyutDrishti: Production-Grade Renewable Energy Forecasting System

## 🚀 Quick Start (Basic Working System)

**Want to see it working immediately?**

1. **Generate Data**:
   ```bash
   python data_pipeline/ingestion/generate_scada_data.py
   ```

2. **Run Basic Forecast**:
   ```bash
   python run_basic_forecast.py
   ```

3. **Launch Interactive Dashboard**:
   ```bash
   python run_dashboard.py
   ```
   Then visit: http://localhost:8503

4. **View Results in Notebook**:
   ```bash
   jupyter notebook notebooks/baseline_forecasting.ipynb
   ```

**Results**: MAE ~0.01 MW, RMSE ~0.01 MW on 35K hourly solar generation records.

---

## Overview

VidyutDrishti is a comprehensive, production-ready renewable energy forecasting system designed for Karnataka SLDC (State Load Dispatch Centre) operations. The system provides probabilistic solar and wind power forecasts with uncertainty quantification, supporting both plant-level and cluster-level predictions.

## Key Features

- **Real-World Scale**: Handles millions of time-series data points
- **Probabilistic Forecasting**: P10, P50, P90 quantiles with uncertainty bands
- **Hybrid Modeling**: Combines tree-based models (LightGBM) with deep learning approaches
- **Explainability**: SHAP-based feature importance and driver analysis
- **Production Architecture**: FastAPI backend, React dashboard, containerized deployment
- **Real Datasets**: NASA POWER, ERA5, NREL data integration
- **Scalability**: Designed for distributed deployment with fault tolerance

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Pipeline │    │  Feature Store  │    │   Model Store   │
│                 │    │                 │    │                 │
│ - Raw Data Ing. │    │ - Time Alignment│    │ - LightGBM      │
│ - Preprocessing │    │ - Feature Eng.  │    │ - LSTM/Transform│
│ - Validation    │    │ - Caching       │    │ - Ensemble      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   Backend       │
                    │                 │
                    │ - /forecast     │
                    │ - /explain      │
                    │ - /health       │
                    │ - Batch API     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │   React         │
                    │   Dashboard     │
                    │                 │
                    │ - Time Series   │
                    │ - Uncertainty   │
                    │ - SHAP Plots    │
                    │ - Alerts        │
                    └─────────────────┘
```

## Project Structure

```
VidyutDrishti/
├── data_pipeline/          # Data ingestion & preprocessing
│   ├── ingestion/         # Raw data download scripts
│   ├── preprocessing/     # Feature engineering
│   └── validation/        # Data quality checks
├── models/                # ML models & training
│   ├── lightgbm/         # Tree-based models
│   ├── deep/             # Neural network models
│   ├── ensemble/         # Model combination
│   └── evaluation/       # Metrics & validation
├── api/                   # FastAPI backend
│   ├── routes/           # API endpoints
│   ├── models/           # Pydantic models
│   └── services/         # Business logic
├── frontend/              # React dashboard
│   ├── src/
│   ├── components/
│   └── public/
├── utils/                 # Shared utilities
│   ├── config/           # Configuration management
│   ├── logging/          # Logging setup
│   └── metrics/          # Custom metrics
├── config/                # Configuration files
├── docker/                # Containerization
└── docs/                  # Documentation
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- Docker (optional)

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd VidyutDrishti
```

2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up Node.js environment
```bash
cd frontend
npm install
```

4. Download real datasets
```bash
cd data_pipeline
python download_nasa_power.py
python download_era5.py
```

5. Train models
```bash
cd models
python train_hybrid_model.py
```

6. Start the system
```bash
# Backend
cd api
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm start
```

## Data Sources

- **NASA POWER**: Global solar irradiance and meteorological data
- **ERA5**: High-resolution weather reanalysis data
- **NREL NSRDB**: Solar resource data for India
- **SCADA-like Generation**: Simulated based on real plant characteristics

## Modeling Approach

### Hybrid Architecture
1. **Global Model**: LightGBM with quantile regression for base forecasts
2. **Local Model**: LSTM for temporal patterns
3. **Ensemble**: Weighted combination with uncertainty calibration

### Uncertainty Quantification
- Quantile regression (P10, P50, P90)
- Ensemble variance estimation
- Conformal prediction intervals

### Feature Engineering
- Temporal features (hour, day, month, season)
- Lagged variables (1h, 24h, 168h)
- Rolling statistics (mean, std, min, max)
- Weather-derived features (clear sky index, wind power density)

## API Endpoints

- `GET /health` - System health check
- `POST /forecast` - Single prediction with uncertainty
- `POST /batch_forecast` - Batch predictions
- `POST /explain` - SHAP explanations for predictions

## Evaluation Metrics

- **nMAE**: Normalized Mean Absolute Error
- **nRMSE**: Normalized Root Mean Square Error
- **Pinball Loss**: Quantile-specific loss
- **CRPS**: Continuous Ranked Probability Score

## Deployment

### Docker
```bash
docker build -t vidyutdrishti .
docker run -p 8000:8000 vidyutdrishti
```

### Kubernetes (Conceptual)
- Data pipeline as CronJob
- Model training as Job
- API as Deployment with HPA
- Frontend as Deployment

## Contributing

1. Follow the established code structure
2. Add tests for new features
3. Update documentation
4. Ensure all CI checks pass

## License

This project is developed for Karnataka SLDC demonstration purposes.