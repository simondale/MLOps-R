from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice.aci import AciWebservice

import datetime
import argparse
import os


def add_arg(parser, name, type, env_var, default=None):
    parser.add_argument(
        name, type=type, default=os.environ.get(env_var, default)
    )


def main(args):
    ws = Workspace.create(
        name=args.workspace_name,
        subscription_id=args.subscription,
        resource_group=args.resource_group,
        location=args.location,
        exist_ok=True,
    )

    environment = Environment.from_docker_image(
        f"{args.environment_name}_server",
        args.docker_image,
        pip_requirements="src/server/requirements.txt"
    )
    environment.inferencing_stack_version = "latest"
    environment.register(ws)

    inference_config = InferenceConfig(
        entry_script="server.py",
        source_directory="src/server",
        environment=environment,
    )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=4
    )

    model = ws.models[args.model_name]
    webservice = Model.deploy(
        ws,
        args.model_name,
        [model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True,
    )

    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting deployment {start}")
    try:
        webservice.wait_for_deployment(
            show_output=True, timeout_sec=args.deploy_timeout
        )
    except Exception as e:
        print(f"Error deploying: {str(e)}")
    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Deployment finished {end}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arg(parser, "--workspace-name", str, "WORKSPACE_NAME")
    add_arg(parser, "--subscription", str, "SUBSCRIPTION_ID")
    add_arg(parser, "--resource-group", str, "RESOURCE_GROUP")
    add_arg(parser, "--location", str, "LOCATION")
    add_arg(parser, "--environment-name", str, "ENVIRONMENT_NAME")
    add_arg(parser, "--deploy-timeout", int, "DEPLOY_TIMEOUT", 1500)
    add_arg(
        parser,
        "--docker-image",
        str,
        "DOCKER_IMAGE",
        "mcr.microsoft.com/mlops/python",
    )
    add_arg(parser, "--model-name", str, "MODEL_NAME")

    args = parser.parse_args()

    main(args)
