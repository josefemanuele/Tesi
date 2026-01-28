""" Utility functions for directory management in experiments."""
from pathlib import Path
from datetime import datetime

# Output diretory structure.
''' data/
      |-- timestamp/
      |    |-- task_name/
      |         |-- model/
      |             |-- model_run.pth
      |         |-- log/
      |             |-- log_run.csv
      |         |-- eval/
      |             |-- log_run.txt
      |         |-- plot/
      |             |-- plot_run.png '''

class DirectoryManager:
    """Class to manage experiment directories."""

    def __init__(self, timestamp: str=None):
        """Initialize with given timestamp."""
        out_folder = "data/"
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
        self.timestamp = timestamp
        self.experiment_folder = out_folder + self.timestamp + "/"

    def get_experiment_folder(self) -> str:
        """Get the experiment folder path and ensure it exists."""
        Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        return self.experiment_folder

    def set_formula_name(self, formula_name: str):
        """Set the formula name and task folder."""
        self.formula_name = formula_name.replace(" ", "_")
        self.task_folder = self.experiment_folder + self.formula_name + "/"

    def get_formula_names(self) -> list:
        """Get list of formula names in the experiment folder."""
        exp_path = Path(self.experiment_folder)
        if not exp_path.exists():
            return []
        formula_names = [p.name for p in exp_path.iterdir() if p.is_dir()]
        return formula_names

    def get_formula_folder(self) -> str:
        """Get the formula folder path and ensure it exists."""
        Path(self.task_folder).mkdir(parents=True, exist_ok=True)
        return self.task_folder

    def get_model_folder(self) -> str:
        """Get the model folder path and ensure it exists."""
        model_folder = self.task_folder + "model/"
        Path(model_folder).mkdir(parents=True, exist_ok=True)
        return model_folder
    
    def get_models(self) -> list:
        """Get list of model files for set formula name."""
        model_folder = self.get_model_folder()
        model_path = Path(model_folder)
        if not model_path.exists():
            return []
        model_files = [p for p in model_path.iterdir() if p.is_file() and p.suffix == ".pth"]
        return model_files

    def get_log_folder(self) -> str:
        """Get the log folder path and ensure it exists."""
        log_folder = self.task_folder + "log/"
        Path(log_folder).mkdir(parents=True, exist_ok=True)
        return log_folder
    
    def get_logs(self) -> list:
        """Get list of log files for the current formula."""
        log_folder = self.task_folder + "log/"
        log_path = Path(log_folder)
        if not log_path.exists():
            return []
        log_files = [p for p in log_path.iterdir() if p.is_file() and p.suffix == ".csv"]
        return log_files

    def get_eval_folder(self) -> str:
        """Get the eval folder path and ensure it exists."""
        eval_folder = self.task_folder + "eval/"
        Path(eval_folder).mkdir(parents=True, exist_ok=True)
        return eval_folder

    def get_plot_folder(self) -> str:
        """Get the plot folder path and ensure it exists."""
        plot_folder = self.task_folder + "plot/"
        Path(plot_folder).mkdir(parents=True, exist_ok=True)
        return plot_folder

