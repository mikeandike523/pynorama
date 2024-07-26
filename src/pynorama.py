import traceback

import click
import termcolor


from analysis import perform_analysis


@click.command()
@click.argument("input_folder", type=click.Path(exists=True))
@click.argument("output_folder", type=click.STRING)
@click.option("--verbose", is_flag=True, default=False)
def main(
    input_folder,
    output_folder,
    verbose=False,
):
    try:
        perform_analysis(input_folder, output_folder)
    except Exception as e:
        tb = traceback.format_exc()
        click.echo(
            termcolor.colored("An error occured during the analysis.", "red"), err=True
        )
        click.echo(termcolor.colored(str(e), "red"), err=True)
        if verbose:
            click.echo("Traceback:", err=True)
            click.echo(tb, err=True)
            exit(1)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
