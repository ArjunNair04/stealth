"""Read-only development tool inventory; Python is optional for native builds."""
import argparse
import json
import os
import platform
import shutil
import subprocess


def inspect(command):
    path = shutil.which(command)
    if path is None:
        return {"available": False}
    flag = "/?" if command.lower().endswith("cl") else "--version"
    try:
        result = subprocess.run([path, flag], capture_output=True, text=True, timeout=15)
        lines = (result.stdout + result.stderr).strip().splitlines()
        return {"available": True, "path": path, "version": lines[0] if lines else "unknown"}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "path": path, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()
    compiler = os.environ.get("CXX", "cl" if platform.system() == "Windows" else "c++")
    report = {
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "tools": {name: inspect(name) for name in ("cmake", "ninja", "git", compiler)},
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['system']} {report['release']} ({report['architecture']})")
        for name, details in report["tools"].items():
            print(f"{name}: {details.get('version', 'MISSING')}")
        print("Native builds require CMake >=3.25, Ninja >=1.11 and a C++20 toolchain.")
    return 0 if all(item["available"] for item in report["tools"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
