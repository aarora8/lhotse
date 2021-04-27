import click

from lhotse.bin.modes import obtain, prepare
from lhotse.recipes.safet import prepare_safet
from lhotse.utils import Pathlike

__all__ = ['safet']


@prepare.command(context_settings=dict(show_default=True))
@click.argument('audio_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('transcripts_dir', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_dir', type=click.Path())
def safet(
        audio_dir: Pathlike,
        transcripts_dir: Pathlike,
        output_dir: Pathlike
):
    """Safet ASR data preparation."""
    prepare_safet(audio_dir, transcripts_dir, output_dir=output_dir)