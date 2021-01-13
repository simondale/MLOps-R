from rpy2.robjects.packages import importr
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
from rpy2.robjects import StrVector

from azureml.core import Run

import os
import argparse


def main(args: argparse.Namespace):
    cran_packages = ["caret", "e1071", "optparse"]
    utils = importr("utils")
    for package in cran_packages:
        utils.install_packages(  # pylint: disable=no-member
            package, quiet=True, repos="http://cran.us.r-project.org"
        )

    with open(args.model_source) as f:
        src = os.linesep.join(f.readlines())
        training = SignatureTranslatedAnonymousPackage(src, "training")

    filename = f"{args.model_name}.rds"
    path = os.path.join(args.model_output, filename)
    os.makedirs(args.model_output, exist_ok=True)

    training.train_model(  # pylint: disable=no-member
        StrVector(["--model-output", path])
    )

    run = Run.get_context()
    run.upload_file(name=filename, path_or_stream=path)
    run.register_model(
        args.model_name, model_path=filename, tags={"type": "R"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="iris-r")
    parser.add_argument("--model-source", type=str, default="model.r")
    parser.add_argument("--model-output", type=str, default="model.rds")
    args = parser.parse_args()
    main(args)
