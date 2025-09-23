"""
油藏数值模拟器安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="reservoir-sim",
    version="0.1.0",
    author="Reservoir Simulation Team",
    author_email="reservoir@example.com",
    description="一个轻量级、模块化的油藏数值模拟器",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/reservoir-sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pyyaml>=5.4.0",
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "viz": [
            "pyvista>=0.32.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reservoir-sim=reservoir_sim.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "reservoir_sim": ["config/*.yaml"],
    },
)
