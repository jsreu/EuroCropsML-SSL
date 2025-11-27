import logging
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import tfparse
import typer
from mlflow.entities import ViewType

from eurocropsssl.experiment.utils import configure_mlflow

logger = logging.getLogger(__name__)

tracking_app = typer.Typer(name="tracking")
configure_mlflow()


@tracking_app.command(help="List tracked experiments.")
def list_experiments(
    active: bool = typer.Option(True, help="List active experiments."),
    deleted: bool = typer.Option(False, help="List deleted experiments."),
    full: bool = typer.Option(
        False,
        "--full",
        help="Show the complete list of experiments. By default only 10 are shown.",
    ),
    filter_string: str = typer.Option(
        "",
        "--filter",
        help="Filter string for selecting experiments. "
        "This follows the same simplified SQL WHERE clause syntax that is used by the "
        "mlflow GUI or search API, see https://mlflow.org/docs/latest/search-runs.html.",
    ),
) -> None:
    """List tracked experiments."""
    if active:
        active_experiments = _list_experiments_as_dataframe(
            ViewType.ACTIVE_ONLY,
            filter_string=filter_string if filter_string else None,
        )
        logger.info(f"Found {len(active_experiments)} matching active experiments.")
        logger.info(
            "\n"
            + active_experiments.sort_values("experiment_id").to_string(
                index=False, max_rows=10 if not full else None
            )
        )

    if deleted:
        deleted_experiments = _list_experiments_as_dataframe(
            ViewType.DELETED_ONLY,
            filter_string=filter_string if filter_string else None,
        )
        logger.info(f"Found {len(deleted_experiments)} matching deleted experiments.")
        logger.info(
            "\n"
            + deleted_experiments.sort_values("experiment_id").to_string(
                index=False, max_rows=10 if not full else None
            )
        )


@tracking_app.command(help="List tracked experiment runs.")
def list_runs(
    experiment_id: str = typer.Option(help="List runs within this experiment."),
    active: bool = typer.Option(True, help="List active runs."),
    deleted: bool = typer.Option(False, help="List deleted runs."),
    full: bool = typer.Option(
        False,
        "--full",
        help="Show the complete list of runs. By default only 10 are shown.",
    ),
    filter_string: str = typer.Option(
        "",
        "--filter",
        help="Filter string for selecting runs. "
        "This follows the same simplified SQL WHERE clause syntax that is used by the "
        "mlflow GUI or search API, see https://mlflow.org/docs/latest/search-runs.html.",
    ),
) -> None:
    """List tracked experiment runts."""
    if active:
        active_runs = _list_experiment_runs_as_dataframe(
            experiment_id,
            ViewType.ACTIVE_ONLY,
            filter_string=filter_string if filter_string else None,
        )
        logger.info(f"Found {len(active_runs)} matching active runs.")
        logger.info(
            "\n"
            + active_runs.sort_values("start_time").to_string(
                index=False, max_rows=10 if not full else None
            )
        )

    if deleted:
        deleted_runs = _list_experiment_runs_as_dataframe(
            experiment_id,
            ViewType.DELETED_ONLY,
            filter_string=filter_string if filter_string else None,
        )
        logger.info(f"Found {len(deleted_runs)} matching deleted runs.")
        logger.info(
            "\n"
            + deleted_runs.sort_values("start_time").to_string(
                index=False, max_rows=10 if not full else None
            )
        )


@tracking_app.command(help="Permanently delete a tracked experiment.")
def delete(
    experiment_id: str = typer.Option(help="Delete this experiment (and all associated runs)."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Make a dry run to determine what would be deleted, "
        "without actually deleting anything yet.",
    ),
) -> None:
    """Permanently delete a tracked experiment and its runs."""
    experiment = mlflow.get_experiment(experiment_id)
    if experiment.lifecycle_stage == "deleted":
        runs = _list_experiment_runs_as_dataframe(
            experiment_id, ViewType.DELETED_ONLY, filter_string=None
        )
        logger.info(f"Found {len(runs)} runs for the experiment with id {experiment_id}.")
        logger.info("\n" + runs.sort_values("start_time").to_string(index=False))
        if dry_run:
            logger.info("These runs would be permanently deleted.")
        elif len(runs) == 0:
            logger.info("Nothing to delete.")
        else:
            confirmed = typer.confirm(
                f"Are you sure you want to permanently delete these {len(runs)} runs."
            )
            if confirmed:
                logger.info(f"{len(runs)} runs will now be permanently deleted.")
                db_config = _db_config_from_terraform(Path("gcloud"))
                try:
                    proxy = subprocess.Popen(["./cloud-sql-proxy", db_config["connection_string"]])
                except FileNotFoundError:
                    logger.warn(
                        "cloud-sql-proxy is required but could not be found. "
                        "Please install it first. For more information, "
                        "see https://cloud.google.com/sql/docs/mysql/sql-proxy#install."
                    )
                    typer.Abort()
                run_ids = ",".join(runs["run_id"].values.tolist())
                subprocess.run(
                    [
                        "mlflow",
                        "gc",
                        "--experiment-ids",
                        experiment_id,
                        "--run-ids",
                        run_ids,
                        "--backend-store-uri",
                        f"postgresql://{db_config['user']}:{db_config['password']}"
                        + f"@localhost:5432/{db_config['database']}",
                    ]
                )
                proxy.terminate()
            else:
                typer.Abort()
    else:
        logger.info(
            f"Can not permanently delete the experiment with id {experiment_id}. "
            "This experiment is still active."
        )


def _list_experiments_as_dataframe(view_type: ViewType, filter_string: str | None) -> pd.DataFrame:
    """Helper to get a dataframe of experiments."""
    column_names = {
        "_experiment_id": "experiment_id",
        "_name": "name",
        "_creation_time": "creation_time",
        "_last_update_time": "last_update_time",
        "_lifecycle_stage": "lifecycle_stage",
        "_tags": "tags",
    }
    experiment_list = mlflow.search_experiments(view_type=view_type, filter_string=filter_string)
    df = pd.DataFrame([vars(exp) for exp in experiment_list], columns=column_names.keys())
    df.rename(columns=column_names, inplace=True)
    df["creation_time"] = pd.to_datetime(df["creation_time"], unit="ms", origin="unix")
    df["last_update_time"] = pd.to_datetime(df["last_update_time"], unit="ms", origin="unix")
    df["experiment_id"] = pd.to_numeric(df["experiment_id"])
    return df


def _list_experiment_runs_as_dataframe(
    experiment_id: str, view_type: ViewType, filter_string: str | None
) -> pd.DataFrame:
    """Helper to get a dataframe of experiment runs."""
    column_names = {
        "_run_id": "run_id",
        "_experiment_id": "experiment_id",
        "_run_name": "run_name",
        "_start_time": "start_time",
        "_end_time": "end_time",
        "_lifecycle_stage": "lifecycle_stage",
    }
    run_list = mlflow.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=view_type,
        filter_string=filter_string,
        output_format="list",
    )
    df = pd.DataFrame([vars(run.info) for run in run_list], columns=column_names.keys())
    df.rename(columns=column_names, inplace=True)
    df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", origin="unix")
    df["end_time"] = pd.to_datetime(df["end_time"], unit="ms", origin="unix")
    df["experiment_id"] = pd.to_numeric(df["experiment_id"])
    return df


def _db_config_from_terraform(path: Path) -> dict[str, str]:
    """Helper to read database configurations from a terraform file."""
    parsed = tfparse.load_from_path(path)

    mlflow_module = [
        module for module in parsed["module"] if module["__tfmeta"]["label"] == "mlflow"
    ][0]
    db_connection_string = ":".join(
        [
            mlflow_module["project"],
            mlflow_module["region"],
            parsed["google_sql_database_instance"][0]["name"],
        ]
    )
    return {
        "user": parsed["google_sql_user"][0]["name"],
        "password": parsed["google_sql_user"][0]["password"],
        "database": parsed["google_sql_database"][0]["name"],
        "connection_string": db_connection_string,
    }
