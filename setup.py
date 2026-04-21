from setuptools import setup, find_packages

setup(
    name="animal-species-prediction",
    version="1.0.0",
    description="Animal species classification using VGG-16 transfer learning on the Animal Image Classification dataset (15 classes)",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow==2.13.0",
        "keras==2.13.1",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "scikit-learn==1.3.0",
        "Pillow==10.0.0",
        "PyYAML==6.0.1",
        "tqdm==4.66.1",
    ],
)
