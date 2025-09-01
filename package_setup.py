from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Parse requirements from requirements.txt
def parse_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="hybrid-xai-healthcare",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Hybrid Explainable AI Models for Personalized Treatment Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hybrid-xai-healthcare",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=parse_requirements(),
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "all": ["pytest", "black", "flake8", "mypy", "sphinx", "sphinx-rtd-theme"],
    },
    entry_points={
        "console_scripts": [
            "hybrid-xai-train=hybrid_xai_healthcare.scripts.train_model:main",
            "hybrid-xai-predict=hybrid_xai_healthcare.scripts.predict:main",
            "hybrid-xai-explain=hybrid_xai_healthcare.scripts.generate_explanations:main",
        ],
    },
)
