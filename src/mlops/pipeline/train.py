from azureml.core import Workspace, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline

import argparse


def main(args: argparse.Namespace):
    ws = Workspace.create(
        name=args.workspace_name,
        subscription_id=args.subscription,
        resource_group=args.resource_group,
        location=args.location,
        exist_ok=True,
    )

    if args.compute_name in ws.compute_targets:
        compute_target = ws.compute_targets[args.compute_name]
    else:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=args.vm_size,
            vm_priority=args.vm_priority,
            min_nodes=args.vm_min_nodes,
            max_nodes=args.vm_max_nodes,
            idle_seconds_before_scaledown=args.vm_idle_seconds,
        )
        compute_target = ComputeTarget.create(
            ws, args.compute_name, compute_config
        )
        compute_target.wait_for_completion(
            show_output=True,
            min_node_count=None,
            timeout_in_minutes=args.vm_timeout_minutes,
        )

    environment = Environment.from_docker_image(
        args.environment_name,
        args.environment_docker_image,
        pip_requirements="src/model/requirements.txt",
    )
    environment.register(ws)

    run_config = RunConfiguration()
    run_config.environment = environment

    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        compute_target=compute_target,
        source_directory="src/model",
        runconfig=run_config,
        allow_reuse=False,
    )

    steps = [train_step]

    train_pipeline = Pipeline(workspace=ws, steps=steps)
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=args.pipeline_name,
        description=args.pipeline_description,
        version=args.version,
    )
    print(published_pipeline)

    pipeline_run = train_pipeline.submit(experiment_name=args.experiment_name)
    print(pipeline_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-name", type=str, default="")
    parser.add_argument(
        "--subscription",
        type=str,
        default="",
    )
    parser.add_argument("--resource-group", type=str, default="")
    parser.add_argument("--location", type=str, default="")
    parser.add_argument("--compute-name", type=str, default="r-train")
    parser.add_argument("--vm-size", type=str, default="STANDARD_D2_V2")
    parser.add_argument("--vm-priority", type=str, default="")
    parser.add_argument("--vm-min-nodes", type=int, default=0)
    parser.add_argument("--vm-max-nodes", type=int, default=1)
    parser.add_argument("--vm-idle-seconds", type=int, default=300)
    parser.add_argument("--vm-timeout-minutes", type=int, default=10)
    parser.add_argument("--environment-name", type=str, default="r-train")
    parser.add_argument(
        "--environment-docker-image",
        type=str,
        default="mcr.microsoft.com/mlops/python",
    )
    parser.add_argument(
        "--pipeline-name", type=str, default="Training Pipeline"
    )
    parser.add_argument(
        "--pipeline-description",
        type=str,
        default="Training pipeline for an R model",
    )
    parser.add_argument("--experiment-name", type=str, default="iris-r")
    parser.add_argument("--version", type=str, default="4")

    args = parser.parse_args()

    main(args)
