# 🌦️ AI in Weather and Tokenomics

## 📋 Overview
This project brings together explainability methods and ML-based weather-forecasting models.


## 🗂️ Folder structure

```
├── analysis/ 📊 : Data analysis and visualization
│   ├── data_loading.py : Data loading utilities
│   ├── main.py : Analysis main runner
│   ├── paths.py : Path configuration
│   └── visualization.py : Plotting and visualization
├── analysis_assets/ 📋 : Analysis configuration files
│   ├── attribution_variables.json : Attribution variable definitions
│   ├── param_to_png.json : Parameter to PNG mapping
│   └── params_registry.json : Parameter registry
├── attribution_pipeline/ 🔍 : Attribution methods and analysis
│   ├── assets.py : Asset management
│   ├── attribution_methods.py : Core attribution algorithms
│   ├── attribution_smoother.py : Smoothing techniques
│   ├── config_loader.py : Configuration loading
│   ├── main.py : Attribution pipeline runner
│   ├── model_wrapper.py : Model integration wrapper
│   ├── utils.py : Utility functions
│   ├── visualize.py : Attribution visualization
│   └── earth2studio/ : Earth2Studio framework integration
│       ├── data/ : Data sources and loaders
│       ├── io/ : Input/output utilities
│       ├── lexicon/ : Data lexicon definitions
│       ├── models/ : Weather forecasting models
│       ├── perturbation/ : Perturbation methods
│       ├── statistics/ : Statistical analysis tools
│       └── utils/ : Framework utilities
├── dashboard/ 🧩 : Streamlit web dashboard
│   ├── app.py : Main dashboard application
│   └── utils.py : Dashboard utilities
├── experiment_assets/ ⚙️ : Experiment configuration files
│   └── config.yaml : Main experiment configuration
├── logs/ 📝 : Experiment logs and outputs
│   └── experiment.log : Experiment log file
├── notebooks/ 📓 : Jupyter notebooks for exploration
│   └── baselines.ipynb : Baselines for attribution methods notebook
├── analysis.sh : Analysis execution script
├── attribution_pipeline.sh : Attribution pipeline execution script
├── dashboard.sh : Dashboard execution script
├── instructions.md : Setup and usage instructions
├── pyproject.toml : Python project configuration
├── uv.lock : UV lock file for dependencies
└── README.md : This file
```

## 🚀 Running experiments

- 🐍 Install python 3.12.
- 🛠️ Install uv
- 🔧 Clone repo, setup environment, run experiments:
    ```
    # clone repo:
    git clone git@github.com:AlainJoss/ai_in_weather_and_tokenomics.git

    # create environment: 
    uv venv --python=3.12
    source .venv/bin/activate
    uv sync --extra fcn --extra fcn3 --extra sfno

    # run 
    sh analysis.sh  # needs attribution tensors files
    sh attribution_pipeline.sh  # needs to be run on cluster, and need baselines files 
    sh dashboard.sh  # needs attribution visualizations files
    ```
- ⚠️ To successfully run the .sh scripts, please ensure you have manually downloaded and placed the required files in the correct paths.