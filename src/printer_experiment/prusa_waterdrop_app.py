import os
import time
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
    
    # You can adjust your guessing limits here!
    min_length: float = Field(default=20.0)
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

    def loop(self, iteration: int) -> None:

        # 1. Ask optimizer and round to a physically printable number (2 decimal places)
        suggested_x = self.opt.ask()
        ridge_length = round(float(suggested_x[0]), 2)
        
        # Keep track of what we actually printed to tell the optimizer later
        executed_x = [ridge_length]

        self.logger.info(f"--- Iteration {iteration + 1} ---")
        console.print(f"Target length: {ridge_length:.2f} mm")

        # Starts Workflow
        workflow = self.workcell_client.start_workflow(
            workflow_definition=self.experiment_workflow,
            json_inputs={
                "length": ridge_length  
            },
            file_inputs={
                "ot2_protocol": str(self.config.protocol_directory / "OT2_CADauto.py")
            }
        )

        console.print("[bold yellow]Executing physical workflow. Waiting for robots to finish...[/bold yellow]")
        
        # --- WAIT FOR REALITY TO CATCH UP ---
        try:
            workflow.wait() 
        except AttributeError:
            while workflow.state.lower() not in ["completed", "failed", "aborted"]:
                time.sleep(10)
                workflow = self.workcell_client.get_workflow(workflow.run_id) 
        # ------------------------------------
        
        console.print("[bold green]Workflow complete! Retrieving image...[/bold green]")

        image_path = self.config.image_directory / f"plate_image_iter_{iteration}.jpg"

        # Explicitly cast image_path to str() and use step_name
        self.data_client.save_datapoint_value( 
            workflow.get_datapoint_id(step_name="Take Picture"), 
            str(image_path),
        )

        console.print("Processing measurement from workflow image...")
        error_distance = camera_driver.get_single_measurement(image_path=str(image_path))

        # Assign a penalty score instead of crashing if OpenCV fails
        if error_distance is None:
            console.print("[bold red]WARNING: OpenCV failed to process the image![/bold red]")
            console.print("Assigning penalty score to optimizer to flag a failed state.")
            error_y = 100.0  
        else:
            error_y = abs(float(error_distance))
            
        # Tell the optimizer the exact rounded value we printed and how far off it was
        self.opt.tell(executed_x, error_y)

        console.print(f"Result: {error_y} mm off.\n")

    def run_experiment(self) -> None:
        console.print("Starting experiment...")

        try:
            for iteration in range(self.config.iterations):
                self.loop(iteration)
                
                # --- MANUAL PAUSE ---
                input("\nAction Required: Please clear the print bed, verify the system is safe, and press [ENTER] to begin the next cycle...\n")

        except Exception as e:
            self.logger.error(f"Experiment stopped: {e}")
            # This will print the exact line of code that caused the crash!
            console.print(traceback.format_exc())

        finally:
            console.print("\nDone")

            if len(self.opt.yi) > 0:
                best_index = np.argmin(self.opt.yi)
                optimal_length = self.opt.Xi[best_index][0]
                lowest_error = self.opt.yi[best_index]

                console.print(f"[bold gold1]Optimal ridge length found:[/bold gold1] {optimal_length:.2f} mm")
                console.print(f"[bold gold1]Minimum error achieved:[/bold gold1] {lowest_error:.2f} mm off-target.")
            else:
                console.print("Experiment failed before any data was recorded.")

if __name__ == "__main__":
    app = PrusaWaterDropExperiment()
    app.run_experiment()
