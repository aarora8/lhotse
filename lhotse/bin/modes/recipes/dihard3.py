import click

from lhotse.bin.modes import obtain, prepare
from lhotse.recipes.dihard3 import prepare_dihard3
from lhotse.utils import Pathlike

__all__ = ["dihard3"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--uem/--no-uem",
    default=True,
    help="Specify whether or not to create UEM supervision",
)
@click.option(
    "-j",
    "-num-jobs",
    type=int,
    default=1,
    help="Number of jobs to scan corpus directory for recordings.",
)
def dihard3(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    uem: Optional[float] = True,
    num_jobs: Optional[int] = 1,
):
    """DIHARD3 data preparation."""
    prepare_dihard3(
        corpus_dir, output_dir=output_dir, uem_manifest=uem, num_jobs=num_jobs
    )
