import os
from invoke import task


@task
def config(context, build_type="Release"):
    print(f"Configuring project with CMake (build_type: {build_type})...")
    build_dir = f"build/{build_type}"
    os.makedirs(build_dir, exist_ok=True)

    if os.name == "nt":
        cmd = [
            "cmake",
            "-G \"MinGW Makefiles\"",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-S",
            ".",
            "-B",
            build_dir
        ]
    else:
        cmd = [
            "cmake",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            "-S",
            ".",
            "-B",
            build_dir
        ]

    print(" ".join(cmd))
    context.run(" ".join(cmd))

    if os.path.islink("compile_commands.json"):
        os.remove("compile_commands.json")

    os.symlink(f"{build_dir}/compile_commands.json", "compile_commands.json")


@task
def build(context, build_type="Release"):
    print("Building project with CMake...")
    build_dir = f"build/{build_type}"

    cmd = [
        "cmake",
        "--build",
        build_dir,
        "-j"
    ]

    print(" ".join(cmd))
    context.run(" ".join(cmd))


@task
def clean(context):
    if os.path.exists("build"):
        if os.name == "nt":
            context.run("rmdir /S /Q build")
        else:
            context.run("rm -rf build")

    if os.path.islink("compile_commands.json"):
        os.remove("compile_commands.json")
