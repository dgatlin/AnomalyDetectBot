import setuptools

setuptools.setup(
    name="anomaly-model",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=["numpy >= 1.18.1", "joblib >= 0.16.0", "scikit-learn >= 0.23.1"],
    python_requires=">=3.7",
)
