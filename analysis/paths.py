from pathlib import Path


class Folders:
    home: Path = Path.home()
    root: Path = Path.cwd()
    analysis: Path = root / "analysis"

    output: Path = analysis / "output"
    plots: Path = analysis / "plots"
    statistics: Path = analysis / "statistics"
    assets: Path = analysis / "analysis_assets"
    data: Path = analysis / "data"

    attributions: Path = data / "attributions_npy"
    predictions: Path = data / "predictions"


class Files:
    parameters: Path = Folders.assets / "params_registry.json"
    variables: Path = Folders.assets / "attribution_variables.json"
