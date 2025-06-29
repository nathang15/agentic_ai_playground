import os
import sys
import asyncio
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")  

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import main

if __name__ == "__main__":
    asyncio.run(main())

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="realinsights-agent",
    version="1.0.0",
    author="Nathan Nguyen",
    author_email="your.email@example.com",
    description="AI Support agent for real estate documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathang15/agentic-ai-playground",
    packages=find_packages(),
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "realinsights-agent=src.cli:main",
        ],
    },
)
