import PyInstaller.__main__
from typing import List

def compile_standalone_executable(source_script_path: str, target_executable_name: str) -> None:
    """compiler_arguments: List[str] = [
        source_script_path,
        f"--name={target_executable_name}",
        "--onefile",
        "--windowed",
        "--hidden-import=tkinter",
        "--hidden-import=numba",
        "--hidden-import=llvmlite",
        "--hidden-import=cv2",
        "--hidden-import=pillow_heif",
        "--collect-all=numba",
        "--collect-all=llvmlite",
        "--collect-all=pillow_heif",
        "--clean"
    ]"""
    compiler_arguments: List[str] = [
        source_script_path,
        f"--name={target_executable_name}",
        "--onefile",
        "--windowed",
        "--hidden-import=tkinter",
        "--hidden-import=numba",
        "--hidden-import=llvmlite",
        "--hidden-import=cv2",
        "--hidden-import=pillow_heif",
        "--collect-all=numba",
        "--collect-all=llvmlite",
        "--collect-all=pillow_heif",
        "--clean"
    ]

    PyInstaller.__main__.run(compiler_arguments)

if __name__ == "__main__":
    compile_standalone_executable("main.py", "PixelSortStudio")