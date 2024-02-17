from setuptools import setup

deps = {
    'std': [
        'pydantic==1.10.8',
        'librosa==0.9.2',
        'numpy==1.24.3',
        'ffmpeg-python>=0.2.0',
        'matplotlib==3.4.3',  # Corrected version number
        'torchvision==0.12.0',
        'torch==1.11.0',
        'dvc==2.58.2',
        'smart-open==6.3.0',
        'mlflow==1.20.0',  # Updated version
        'optuna==3.3.0',
        'onnx==1.15.0',
        'onnxruntime==1.17.0'
    ]
}

setup(
    name='music-genre-classification',
    version='1.0.1',
    description='Music Genre Classification',
    install_requires=[],
    extras_require=deps,
    entry_points={
        'console_scripts': [
            'pipeline = __main__.py:cli'
        ]
    },
    packages=['genre_classification']  # Add your package name here
)
