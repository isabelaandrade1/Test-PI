from setuptools import setup, find_packages

setup(
    name="agiliza-pi",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "Flask==2.0.1",
        "tensorflow==2.17.0",
        "scikit-learn==1.0.2",
        "pandas==1.2.4",
        "gunicorn==20.1.0",
        "cython==0.29.24",
        "scipy==1.5.4",
        "wheel==0.43.0",
        "werkzeug==2.0.3"
    ],
)
