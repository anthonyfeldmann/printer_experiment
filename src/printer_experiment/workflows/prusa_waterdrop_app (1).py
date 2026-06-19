import os
from pathlib import Path
from typing import Optional
import numpy as np
from skopt import Optimizer

from madsci.client import WorkcellClient
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
   
    iterations: int = Field(default=10, gt=0)
    min_length: float = Field(default=10.0)
    max_length: float = Field(default=100.0)


class PrusaWaterDropExperiment(ExperimentApplication):
    config = PrusaWaterDropConfig()

    def __init__(self, config: Optional[PrusaWaterDropConfig] = None):
        if config:
            self.config = config
        super().__init__()
       
        self.workcell_client = WorkcellClient("http://localhost:8005")
       
        self.experiment_workflow = WorkflowDefinition.from_yaml(
            self.config.workflow_directory / "autonomous_drop_workflow.yaml"
        )

        self.opt = Optimizer(
            dimensions=[(self.config.min_length, self.config.max_length)],
            base_estimator="GP",
            acq_func="EI",
            n_initial_points=3,
            random_state=237
        )

    def loop(self, iteration: int) -> None:
       
        #optimizer picks the next ridge length
        suggested_x = self.opt.ask()
        ridge_length = float(suggested_x[0])
       
        self.logger.info(f"--- Iteration {iteration + 1} ---")
        console.print(f"target length: {ridge_length:.2f} mm")

        #Starts Workflow (Prusa prints, then OT-2 drops)
        workflow = self.workcell_client.start_workflow(
            workflow_definition=self.experiment_workflow,
            json_inputs={
                "prusa_args": {"length": ridge_length}
            },
            file_inputs={
                "ot2_protocol": str(self.config.protocol_directory / "OT2_CADauto.py")
            }
        )
        self.data_client.save_datapoint_value(  # type: ignore[attr-defined]
            workflow.get_datapoint_id(step_key="take_picture"),
            self.config.image_directory / "plate_image.jpg",
        )

       
        #measures value
        console.print("taking measurment")
        error_distance = camera_driver.get_single_measurement()
       
        if error_distance is None:
             raise RuntimeError("OpenCV failed")
       
        #optimizer
        error_y = abs(float(error_distance))
        self.opt.tell(suggested_x, error_y)
       
        console.print(f" Result: {error_y} mm off.\n")

    def run_experiment(self) -> None:
        console.print("starting experiment")
       
        try:
            for iteration in range(self.config.iterations):
                self.loop(iteration)
               
        except Exception as e:
            self.logger.error(f"Experiment stopped: {e}")
           
        finally:
            console.print("\n Done")
           
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