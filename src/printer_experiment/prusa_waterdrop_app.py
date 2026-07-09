import os
import json
import traceback
from pathlib import Path
from typing import Optional
import numpy as np
from skopt import Optimizer

from madsci.client import WorkcellClient, DataClient
from madsci.common.types.base_types import PathLike
from madsci.experiment_application import ExperimentApplication
from madsci.experiment_application.experiment_application import ExperimentApplicationConfig
from madsci.common.types.workflow_types import WorkflowDefinition
from pydantic import Field
from rich.console import Console

import camera_driver

console = Console()

class PrusaWaterDropConfig(ExperimentApplicationConfig):
    workflow_directory: PathLike = (Path(__file__).parent / "workflows").resolve()
    protocol_directory: PathLike = (Path(__file__).parent / "protocols").resolve()
    image_directory: PathLike = (Path(__file__).parent / "images").resolve()
    update_node_files: bool = False
    
    iterations: int = Field(default=10, gt=0)
    min_length: float = Field(default=10.0)
    max_length: float = Field(default=60.0)

class PrusaWaterDropExperiment(ExperimentApplication):
    config = PrusaWaterDropConfig()

    def __init__(self, config: Optional[PrusaWaterDropConfig] = None):
        if config:
            self.config = config
        super().__init__()

        self.workcell_client = WorkcellClient("http://parker.cels.anl.gov:8005")
        self.data_client = DataClient("http://parker.cels.anl.gov:8003")

        yaml_path = self.config.workflow_directory / "autonomous_drop_workflow.yaml"
        console.print(f"[bold green]LOADING YAML FROM:[/bold green] {yaml_path}")

        self.experiment_workflow = WorkflowDefinition.from_yaml(yaml_path)

        self.opt = Optimizer(
            dimensions=[(self.config.min_length, self.config.max_length)],
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=3,
            random_state=237
        )

        self.config.image_directory.mkdir(parents=True, exist_ok=True)
        
        # --- THE PERSISTENT MEMORY INIT ---
        self.memory_file = self.config.workflow_directory / "optimizer_memory.json"
        self.history_x = []
        self.history_y = []

        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    memory_data = json.load(f)
                
                if memory_data.get("ridge_lengths") and memory_data.get("skopt_y"):
                    self.history_x = [[x] for x in memory_data["ridge_lengths"]]
                    self.history_y = memory_data["skopt_y"]
                    
                    self.opt.tell(self.history_x, self.history_y)
                    console.print(f"[bold cyan]Loaded {len(self.history_y)} previous physical runs from memory.[/bold cyan]")
            except Exception as e:
                console.print(f"[bold red]Failed to load memory file: {e}[/bold red]")

    def loop(self, iteration: int) -> None:
        suggested_x = self.opt.ask()
        ridge_length = round(float(suggested_x[0]), 2)
        executed_x = [ridge_length]

        self.logger.info(f"--- Iteration {iteration + 1} ---")
        console.print(f"Target length: {ridge_length:.2f} mm")

        # Starts Workflow on the physical hardware
        workflow = self.workcell_client.start_workflow(
            workflow_definition=self.experiment_workflow,
            json_inputs={
                "length": ridge_length  
            },
            file_inputs={
                "ot2_protocol": str(self.config.protocol_directory / "OT2_CADauto.py")
            }
        )

        image_path = self.config.image_directory / f"plate_image_iter_{iteration}.jpg"

        self.data_client.save_datapoint_value(  # type: ignore[attr-defined]
            workflow.get_datapoint_id(step_key="take_picture"),
            image_path,
        )

        console.print("Processing measurement from workflow image...")
        
        total_score = camera_driver.get_single_measurement(image_path=str(image_path), target_bucket=1)

        if total_score is None:
             raise RuntimeError("OpenCV failed to process the image")

        console.print(f"Result: {total_score:.2f} mm fluid score.\n")

        # --- THE DATA GATEKEEPER ---
        user_choice = input("Accept and save this data point? (y/n): ").strip().lower()

        if user_choice == 'y':
            skopt_y = -float(total_score)
            self.opt.tell(executed_x, skopt_y)
            
            self.history_x.append(executed_x)
            self.history_y.append(skopt_y)

            try:
                with open(self.memory_file, 'w') as f:
                    json.dump({
                        "ridge_lengths": [x[0] for x in self.history_x],
                        "skopt_y": self.history_y,
                        "human_scores": [-y for y in self.history_y] 
                    }, f, indent=4)
                console.print("[bold green]Data saved to memory and optimizer.[/bold green]")
            except Exception as e:
                console.print(f"[bold red]Failed to save to memory file: {e}[/bold red]")
        else:
            console.print("[bold yellow]Data discarded. The optimizer will ignore this physical trial.[/bold yellow]")

        # --- THE PHYSICAL RESET PAUSE ---
        input("\nAction Required: Please clear the print bed, verify the system is safe, and press [ENTER] to begin the next cycle...\n")

    def run_experiment(self) -> None:
        console.print("Starting experiment...")

        try:
            for iteration in range(self.config.iterations):
                self.loop(iteration)
                
        except Exception as e:
            self.logger.error(f"Experiment stopped: {e}")
            console.print(traceback.format_exc())

        finally:
            console.print("\nDone")

            if len(self.opt.yi) > 0:
                best_index = np.argmin(self.opt.yi)
                optimal_length = self.opt.Xi[best_index][0]
                best_score = -self.opt.yi[best_index]

                console.print(f"[bold gold1]Optimal ridge length found:[/bold gold1] {optimal_length:.2f} mm")
                console.print(f"[bold gold1]Maximum liquid score achieved:[/bold gold1] {best_score:.2f} mm")
            else:
                console.print("Experiment failed before any data was recorded.")

if __name__ == "__main__":
    app = PrusaWaterDropExperiment()
    app.run_experiment()
