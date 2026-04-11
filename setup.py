from setuptools import find_packages, setup

setup(
    name="mlops_e2e",
    version="0.2.0",
    description="MLOps E2E — California Housing regression + 20 Newsgroups ensemble classifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        # Housing regression pipeline
        "lightgbm>=4.0.0",
        "optuna>=3.0.0",
        "optuna-integration[mlflow]>=3.0.0",
        "mlflow>=2.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        # Newsgroups ensemble pipeline
        "gensim>=4.3.0",
        "torch>=2.1.0",
        "pytorch-lightning>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pyspark>=3.5.0",
            "ruff>=0.8.0",
            "fastapi>=0.104.0",
            "httpx>=0.25.0",
        ],
    },
)
