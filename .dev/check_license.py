# BSD 3-Clause License
#
# Copyright (c) 2021-2024, Massachusetts Institute of Technology.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Script to work with licenses."""

import difflib
import pathlib

import click


def get_filetype(input_file):
    """Get filetype based on extension."""
    extension = pathlib.Path(input_file.name).suffix
    if extension in [".cpp", ".h"]:
        return "cpp"
    elif extension == ".py":
        return "python"
    else:
        return None


def get_license(filename="LICENSE"):
    """Get license text."""
    pkg_path = pathlib.Path(__file__).absolute().parent.parent
    with (pkg_path / filename).open("r") as fin:
        return fin.read()


def _gen_cpp_license(license, width=80):
    prefix = "/* " + "-" * (width - 3) + "\n"
    suffix = " * " + "-" * (width - 6) + " */\n"
    lines = license.split("\n")
    new_lines = [" * " + x if len(x) > 0 else " *" for x in lines]
    return prefix + "\n".join(new_lines) + suffix


def _gen_py_license(license):
    return "\n".join(["# " + x if len(x) > 0 else "#" for x in license.split("\n")])


def _has_license(input_file, target):
    filetype = get_filetype(input_file)
    num_lines = len(target.split("\n")) - 1
    try:
        lines = [next(input_file) for _ in range(num_lines)]
    except StopIteration:
        return False, None

    if filetype == "python":
        if lines and "#!" in lines[0]:
            lines = lines[1:]
            try:
                lines.append(next(input_file))
            except StopIteration:
                return False, None

    header = "".join(lines)
    if header != target:
        diff = difflib.context_diff(
            header.splitlines(keepends=True),
            target.splitlines(keepends=True),
            fromfile=input_file.name,
            tofile="license",
        )
        diff_str = "".join(diff)
        return False, diff_str

    return True, None


@click.group()
def main():
    """Tool to check or add license headers."""
    pass


@main.command()
@click.argument("input_files", type=click.File("r"), nargs=-1)
@click.option("--verbose", "-v", is_flag=True, help="show files checked")
def check(input_files, verbose):
    """Check if header of file matches license."""
    license = get_license()
    formatters = {"python": _gen_py_license, "cpp": _gen_cpp_license}

    for input_file in input_files:
        filetype = get_filetype(input_file)
        if not filetype:
            click.secho(f"unknown extension for {input_file.name}!", fg="yellow")
            continue

        exists, diff = _has_license(input_file, formatters[filetype](license))
        if not exists:
            if diff is None:
                click.secho(f"{input_file.name}: missing", fg="red")
            else:
                click.secho(f"{input_file.name}: bad", fg="red")
                if verbose:
                    click.secho(f"{diff}", fg="red")
        elif verbose:
            click.secho(f"{input_file.name}: good", fg="green")


@main.command()
@click.argument("input_files", type=click.File("r"), nargs=-1)
def add(input_files):
    """Check if header of file matches license."""
    license = get_license()
    formatters = {"python": _gen_py_license, "cpp": _gen_cpp_license}

    for input_file in input_files:
        filetype = get_filetype(input_file)
        if not filetype:
            click.secho(f"unknown extension for {input_file.name}!", fg="yellow")
            continue

        text = formatters[filetype](license)
        exists, _ = _has_license(input_file, text)
        if exists:
            continue

        input_file.seek(0)
        lines = [x.strip("\n") for x in input_file]
        license_lines = text.split("\n")
        if filetype == "python" and lines and "#!" in lines[0]:
            all_lines = [lines[0]] + license_lines + lines[1:]
        else:
            all_lines = license_lines + lines

        with pathlib.Path(input_file.name).open("w") as fout:
            fout.write("\n".join(all_lines) + "\n")


if __name__ == "__main__":
    main()
