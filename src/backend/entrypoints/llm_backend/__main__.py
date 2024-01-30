import click

from market_alerts.entrypoints.llm_backend.app import (
    run_api,
    run_celery,
    run_celery_beat,
)


@click.group()
def cli() -> None:
    pass


@click.command()
def serve() -> None:
    run_api()


@click.command()
def celery() -> None:
    run_celery()


@click.command("celery_beat")
def celery_beat() -> None:
    run_celery_beat()


if __name__ == "__main__":
    cli.add_command(serve)
    cli.add_command(celery)
    cli.add_command(celery_beat)
    cli()
