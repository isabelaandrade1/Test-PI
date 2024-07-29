from setuptools import setup, find_packages

setup(
    name="agiliza-pi",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "Flask==2.0.1",
        "tensorflow==2.17.0",
        "scikit-learn==1.5.1",
        "pandas==2.2.2",
        "gunicorn==20.1.0",
        "Cython==3.0.10",
        "scipy==1.5.4",
        "wheel==0.43.0",
        "Werkzeug==2.0.3"
    ],
)
