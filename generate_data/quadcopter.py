from functools import partial
from math import pi
from pathlib import Path
import dymos as dm
import multiprocessing
import numpy as np
import openmdao.api as om
import os
import random
import shutil

"""
Quadcopter Landing: Data Generation.

This Python script uses Dymos to generate energy-optimized trajectories for
a quadcopter landing problem.

Please see the top-level project README.md for information on how to run it.
"""

#---- Constants ---------------------------------------------------------------

# Number of processes to use for data generation.
N_PROCESSES = int(os.getenv("GENERATE_DATA_N_PROCESSES", 8))

# Number of trajectories to generate.
N_TRAJECTORIES = int(os.getenv("GENERATE_DATA_N_TRAJECTORIES", 15000))

# Current project directory (based on the location of this file).
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).resolve()

# Quadcopter data directory.
DATA_DIR = PROJECT_DIR / "../trajectories/quadcopter"


#------------------------------------------------------------------------------


class QuadcopterODE(om.ExplicitComponent):
    
    """Dymos representation of the Quadcopter ODE."""

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # Inputs: vx, vz, theta, thrust (u1), omega (u2)
        self.add_input("vx", val=np.zeros(nn), desc="horizontal velocity", units="m/s")
        self.add_input("vz", val=np.zeros(nn), desc="vertical velocity", units="m/s")
        self.add_input(
            "theta", val=np.ones(nn), desc="angle of quadcopter", units="rad"
        )
        self.add_input(
            "thrust", val=np.ones(nn), desc="thrust of quadcopter", units="N"
        )
        self.add_input(
            "omega", val=np.ones(nn), desc="rotation rate of quadcopter", units="rad/s"
        )

        # Outputs: xdot, zdot, vxdot, vzdot, thetadot, energydot
        self.add_output(
            "xdot",
            val=np.zeros(nn),
            desc="horizontal velocity",
            units="m/s",
            tags=["dymos.state_rate_source:x", "dymos.state_units:m"],
        )
        self.add_output(
            "zdot",
            val=np.zeros(nn),
            desc="vertical velocity",
            units="m/s",
            tags=["dymos.state_rate_source:z", "dymos.state_units:m"],
        )
        self.add_output(
            "vxdot",
            val=np.zeros(nn),
            desc="horizontal acceleration",
            units="m/s/s",
            tags=["dymos.state_rate_source:vx", "dymos.state_units:m/s"],
        )
        self.add_output(
            "vzdot",
            val=np.zeros(nn),
            desc="vertical acceleration",
            units="m/s/s",
            tags=["dymos.state_rate_source:vz", "dymos.state_units:m/s"],
        )
        self.add_output(
            "thetadot",
            val=np.zeros(nn),
            desc="rotational velocity",
            units="rad/s",
            tags=["dymos.state_rate_source:theta", "dymos.state_units:rad"],
        )
        self.add_output(
            "energydot",
            val=np.zeros(nn),
            desc="rate of energy use",
            tags=["dymos.state_rate_source:energy"],
        )

        # Setup partial derivatives
        partials_range = np.arange(self.options["num_nodes"])
        self.declare_partials(
            of="xdot", wrt="vx", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="zdot", wrt="vz", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="vxdot", wrt="thrust", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="vxdot", wrt="theta", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="vzdot", wrt="thrust", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="vzdot", wrt="theta", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="thetadot", wrt="omega", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="energydot", wrt="thrust", rows=partials_range, cols=partials_range
        )
        self.declare_partials(
            of="energydot", wrt="omega", rows=partials_range, cols=partials_range
        )

    def compute(self, inputs, outputs):
        vx = inputs["vx"]
        vz = inputs["vz"]
        theta = inputs["theta"]
        thrust = inputs["thrust"]
        omega = inputs["omega"]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        outputs["xdot"] = vx
        outputs["zdot"] = vz
        outputs["vxdot"] = thrust * sin_theta
        outputs["vzdot"] = thrust * cos_theta - 9.81
        outputs["thetadot"] = omega
        outputs["energydot"] = thrust * thrust + omega * omega

    def compute_partials(self, inputs, partials):
        theta = inputs["theta"]
        thrust = inputs["thrust"]
        omega = inputs["omega"]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        partials["xdot", "vx"] = 1
        partials["zdot", "vz"] = 1
        partials["vxdot", "thrust"] = sin_theta
        partials["vxdot", "theta"] = thrust * cos_theta
        partials["vzdot", "thrust"] = cos_theta
        partials["vzdot", "theta"] = -thrust * sin_theta
        partials["thetadot", "omega"] = 1
        partials["energydot", "thrust"] = 2 * thrust
        partials["energydot", "omega"] = 2 * omega


def check_partial_derivatives():
    """
    Check the provided partial derivatives in the QuadcopterODE model against
    numerical approximations.
    """

    num_nodes = 5

    p = om.Problem(model=om.Group())

    # Outputs of the IVC are inputs to the model
    ivc = p.model.add_subsystem("vars", om.IndepVarComp())
    ivc.add_output("vx", shape=(num_nodes,), units="m/s")
    ivc.add_output("vz", shape=(num_nodes,), units="m/s")
    ivc.add_output("theta", shape=(num_nodes,), units="rad")
    ivc.add_output("thrust", shape=(num_nodes,), units="N")
    ivc.add_output("omega", shape=(num_nodes,), units="rad/s")

    p.model.add_subsystem("ode", QuadcopterODE(num_nodes=num_nodes))

    p.model.connect("vars.vx", "ode.vx")
    p.model.connect("vars.vz", "ode.vz")
    p.model.connect("vars.theta", "ode.theta")
    p.model.connect("vars.thrust", "ode.thrust")
    p.model.connect("vars.omega", "ode.omega")

    p.setup(force_alloc_complex=True)

    p.set_val("vars.vx", 10 * np.random.random(num_nodes))
    p.set_val("vars.vz", 10 * np.random.random(num_nodes))
    p.set_val("vars.theta", 10 * np.random.random(num_nodes))
    p.set_val("vars.thrust", 10 * np.random.random(num_nodes))
    p.set_val("vars.omega", 10 * np.random.random(num_nodes))

    p.run_model()
    cpd = p.check_partials(method="cs", compact_print=True)


def solve_single_trajectory(
    parent_dir: Path,
    name: str,
    x0: float,
    z0: float,
    vx0: float,
    vz0: float,
    theta0: float,
) -> om.Problem:
    """
    Solve for a single trajectory given the supplied starting conditions.

    - `parent_dir` should be populated to safely remove the reports and
      databases generated when running a solution.
    - `name` is the name of the problem for report file generation.

    This returns a `Problem` object. To retrieve a value from the `Problem`,
    refer to it by its string name:

        >>> exp_out = solve_single(-5, 5, 0, 0, 0)
        >>> ts = exp_out.get_val("traj.phase0.timeseries.time")

    Valid string names (that I know of) are `time`, all the state variables,
    and the controls.
    """
    # Initialize the problem and the optimization driver
    p = om.Problem(model=om.Group(), name=name)
    p.driver = om.ScipyOptimizeDriver()
    p.driver.declare_coloring()

    # Create a trajectory and add a phase to it
    traj = p.model.add_subsystem("traj", dm.Trajectory())
    phase = traj.add_phase(
        "phase0",
        dm.Phase(
            ode_class=QuadcopterODE, transcription=dm.GaussLobatto(num_segments=10)
        ),
    )

    # Set the variables
    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10))
    phase.add_state("energy", fix_initial=True, fix_final=False)
    phase.add_state("x", fix_initial=True, fix_final=True)
    phase.add_state("z", fix_initial=True, fix_final=True)
    phase.add_state("theta", fix_initial=True, fix_final=True)
    phase.add_state("vx", fix_initial=True, fix_final=True)
    phase.add_state("vz", fix_initial=True, fix_final=True)
    phase.add_control(
        "thrust",
        continuity=True,
        rate_continuity=True,
        units="N",
        lower=0.0,
        upper=20.0,
    )
    phase.add_control(
        "omega",
        continuity=True,
        rate_continuity=True,
        units="rad/s",
        lower=-2.0,
        upper=2.0,
    )

    # Minimised the energy used for landing
    phase.add_objective("energy", loc="final")

    p.setup()

    # Set the initial values
    phase.set_time_val(initial=0.0, duration=10.0)
    phase.set_state_val("energy", [0, 800])
    phase.set_state_val("x", [x0, 0])
    phase.set_state_val("z", [z0, 0.1])
    phase.set_state_val("theta", [theta0, 0])
    phase.set_state_val("vx", [vx0, 0])
    phase.set_state_val("vz", [vz0, 0])
    phase.set_control_val("thrust", [20, 0])
    phase.set_control_val("omega", [0, 0])

    # Solve for the trajectory
    dm.run_problem(p, solution_record_file=f"{name}.db")
    exp_out = traj.simulate()

    # Remove the report and simulation database.
    shutil.rmtree(parent_dir / f"./reports/{name}")
    (parent_dir / f"./{name}.db").unlink()

    return exp_out


def solve_single_trajectory_and_deduplicate(
    parent_dir: Path,
    name: str,
    x0: float,
    z0: float,
    vx0: float,
    vz0: float,
    theta0: float,
) -> np.ndarray:
    """
    Solve for a single trajectory given the supplied starting conditions.

    - `parent_dir` should be populated to safely repove the reports and
      databased generated when running a solution.
    - `name` is the name of the problem for report file generation.

    This de-duplicates time values from the returned array.

    This returns a numpy array with the following columns:
       - time
       - x
       - z
       - theta
       - vx
       - vz
       - thrust
       - omega
    """
    # Solve the trajectory
    sim = solve_single_trajectory(parent_dir, name, x0, z0, vx0, vz0, theta0)

    # Collect time, the state variables, and the control signels
    ts_name = "traj.phase0.timeseries"

    def get_val(col_name):
        col = sim.get_val(f"{ts_name}.{col_name}")
        assert col is not None
        return col

    column_names = ["time", "x", "z", "theta", "vx", "vz", "thrust", "omega"]
    column_values = tuple(map(get_val, column_names))
    array_with_duplicates = np.hstack(column_values)

    # De-duplicate the times in the array
    _, unique_indices = np.unique(array_with_duplicates[:, 0], return_index=True)
    array = np.asarray(array_with_duplicates[unique_indices], dtype=np.float32)

    return array


def solve_random_trajectory(parent_dir: Path, seed: int) -> np.ndarray:
    """
    Pick a random starting point, solve its trajectory, and return its
    simulated run.

    - `parent_dir` should be populated to safely remove the reports and
      databased generated when running a solution.

    This returns a numpy array with the following columns:
       - time
       - x
       - z
       - theta
       - vx
       - vz
       - thrust
       - omega
    """
    # Seed the random number generators.
    rng = random.Random(seed)
    np.random.seed(rng.randint(0, 2**32 - 1))

    # Generate random starting points for the trajectory.
    x0 = rng.uniform(-5, 5)
    z0 = rng.uniform(5, 20)
    vx0 = rng.uniform(-1, 1)
    vz0 = rng.uniform(-1, 1)
    theta0 = rng.uniform(-pi / 10, pi / 10)

    # Run the simulation.
    name = f"quadcopter_{seed:07}"
    return solve_single_trajectory_and_deduplicate(
        parent_dir, name, x0, z0, vx0, vz0, theta0
    )


def ensure_trajectory(parent_dir: Path, data_dir: Path, trajectory_number: int):
    """
    Ensure that the supplied trajectory number has been written to an npz
    file in the supplied parent directory.

    If the file already exists, this returns doing nothing. If the file
    does not exist, the trajectory is simulated and the npz file is saved.
    """
    out_file_path = data_dir / f"{trajectory_number:07}.npz"
    if not out_file_path.exists():
        array = solve_random_trajectory(parent_dir, trajectory_number)
        np.savez_compressed(out_file_path, trajectory=array)


def multiprocess_create_trajectories(
    parent_dir: Path, data_dir: Path, n_trajectories: int
):
    """
    Run the `ensure_trajectory` operation in multiple processes to generate
    trajectories.
    """
    worker_fn = partial(ensure_trajectory, parent_dir, data_dir)
    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        pool.map(worker_fn, range(n_trajectories))


def main():
    # Run multi-process data generation
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    multiprocess_create_trajectories(PROJECT_DIR, DATA_DIR, N_TRAJECTORIES)

    # Remove extraneous stuff generated during the solution processes
    (PROJECT_DIR / "openmdao_checks.out").unlink(missing_ok=True)
    shutil.rmtree(PROJECT_DIR / "reports", ignore_errors=True)
    shutil.rmtree(PROJECT_DIR / "coloring_files", ignore_errors=True)


if __name__ == "__main__":
    main()
