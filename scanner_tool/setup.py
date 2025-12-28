from setuptools import setup, find_packages

setup(
    name="scanner_tool",
    version="0.1.0",
    description="PC 端 3D 场景处理工具",
    author="Scanner Tool Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.6.0",
        "pillow>=8.1.0",
        "opencv-python>=4.1",
        "open3d>=0.12.0",
        "scikit-video>=1.1.11",
        "torch>=2.0",
        "torchvision>=0.15",
        "tqdm",
        "rich",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "scanner-tool=scanner_tool.cli.main:main",
        ],
    },
)
