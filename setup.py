from setuptools import setup, find_packages

deps = [
    'pydantic==1.10.8',
    'librosa==0.9.2',
    'numpy==1.24.3',
    'ffmpeg-python>=0.2.0',
    'matplotlib==3.7.1',
    'torchvision==0.12.0',
    'torch==1.11.0',
    'dvc==2.58.2',
    'smart-open==6.3.0',
    'mlflow==2.6.0',
    'optuna==3.3.0'

]

setup(name='music-genre-classification',
      version='1.0.1',
      description='Music Genre Classification',
      install_requires=[],
      extras_require=deps,
      include_package_data=True,
      tests_require=[
      ],
      setup_requires=[
      ],
      entry_points={
          'console_scripts': [
              'pipeline = __main__.py:cli'
          ]
      }
      )
