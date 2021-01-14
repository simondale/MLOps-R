from azureml.core import Workspace, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData

import argparse
import os


def add_arg(parser, name, type, env_var, default=None):
    parser.add_argument(
        name, type=type, default=os.environ.get(env_var, default)
    )


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

    pipeline_data = PipelineData(
        args.experiment_name, ws.get_default_datastore()
    )

    train_step = PythonScriptStep(
        name="Train Model",
        script_name="train.py",
        compute_target=compute_target,
        source_directory="src/model",
        runconfig=run_config,
        allow_reuse=False,
        arguments=["--model-output", pipeline_data],
        outputs=[pipeline_data],
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

    pipeline_run.wait_for_completion(
        show_output=True, timeout_seconds=args.pipeline_timeout
    )

    if pipeline_run.get_status() == "Finished":
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg(parser, "--workspace-name", str, "WORKSPACE_NAME")
    add_arg(parser, "--subscription", str, "SUBSCRIPTION_ID")
    add_arg(parser, "--resource-group", str, "RESOURCE_GROUP")
    add_arg(parser, "--location", str, "LOCATION")
    add_arg(parser, "--compute_name", str, "COMPUTE_NAME")
    add_arg(parser, "--vm-size", str, "VM_SIZE")
    add_arg(parser, "--vm-priority", str, "VM_PRIORITY", "")
    add_arg(parser, "--vm-min-nodes", int, "VM_MIN_NODES", 0)
    add_arg(parser, "--vm-max-nodes", int, "VM_MAX_NODES", 1)
    add_arg(parser, "--vm-idle-seconds", int, "VM_IDLE_SECONDS", 300)
    add_arg(parser, "--vm-timeout-minutes", int, "VM_TIMEOUT_MINUTES", 10)
    add_arg(parser, "--environment-name", str, "ENVIRONMENT_NAME")
    add_arg(
        parser,
        "--environment-docker-image",
        str,
        "ENVIRONMENT_DOCKER_IMAGE",
        "mcr.microsoft.com/mlops/python",
    )
    add_arg(
        parser, "--pipeline-name", str, "PIPELINE_NAME", "Training Pipeline"
    )
    add_arg(
        parser,
        "--pipeline-description",
        str,
        "PIPELINE_DESCRIPTION",
        "Training pipeline for an R model",
    )
    add_arg(parser, "--pipeline-timeout", int, "PIPELINE_TIMEOUT", 1500)
    add_arg(parser, "--experiment-name", str, "EXPERIMENT_NAME")
    add_arg(parser, "--version", str, "MODEL_VERSION", "6")

    args = parser.parse_args()

    main(args)
