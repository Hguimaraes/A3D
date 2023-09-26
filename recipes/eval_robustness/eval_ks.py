import sys
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from a3d import KWSBrain
from a3d.dataset import prep_speechcommands
from a3d.dataset import create_datasets_speechcommands

def main(hparams, hparams_file, run_opts, overrides):
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data prep to run on the main thread
    sb.utils.distributed.run_on_main(
        prep_speechcommands,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["annotation_folder"]
        },
    )

    # Create dataset objects
    datasets, label_encoder = create_datasets_speechcommands(hparams)

    # Initialize the Trainer
    brain = KWSBrain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts
    )

    # Evaluate the best checkpoint on the test set
    brain.evaluate_robustness(
        test_set=datasets["test"],
        test_loader_kwargs=hparams["dataloader_options"]
    )


if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    main(hparams, hparams_file, run_opts, overrides)