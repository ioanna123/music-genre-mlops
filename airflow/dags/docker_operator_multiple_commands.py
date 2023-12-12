from datetime import datetime

from airflow.decorators import task, dag
from airflow.providers.docker.operators.docker import DockerOperator


@dag(start_date=datetime(2021, 1, 1), schedule_interval='@daily', catchup=False)
def docker_dag_multiple_commands():
    @task
    def t1():
        pass

    t2 = DockerOperator(
        task_id='t2',
        image='music-genre-classif:latest',
        # command="bash -c 'ls -l checkpoints && dvc add checkpoints/resnet18_checkpoint.pt'",
        command="bash -c 'dvc add checkpoints/resnet18_checkpoint.pt && dvc push && git add checkpoints/resnet18_checkpoint.pt.dvc'",
        # command='python3 __main__.py train-using-image-features --model resnet18 --criterion cross_entropy --optimizer sdg --checkpoints_path checkpoints --images_path Data/images_original && ls -l checkpoints',
        network_mode='bridge',
        environment={
            'AWS_DEFAULT_REGION': 'us-east-1',
            'AWS_REGION': 'us-east-1',
            'AWS_SECRET_ACCESS_KEY': 'add_the_AWS_SECRET_ACCESS_KEY',
            'AWS_ACCESS_KEY_ID': 'add_the_AWS_ACCESS_KEY_ID',
            'AWS_SESSION_TOKEN': 'add_the_AWS_SESSION_TOKEN'

        }
    )

    t1() >> t2


dag = docker_dag_multiple_commands()
