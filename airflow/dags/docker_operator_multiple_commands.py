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
            'AWS_SECRET_ACCESS_KEY': 'SoWpDJzMmRsd+gn1vvv1prrmDxeg6qG3aCr6/iS+',
            'AWS_ACCESS_KEY_ID': 'ASIAXPJZLAWLCKT75S6C',
            'AWS_SESSION_TOKEN': 'IQoJb3JpZ2luX2VjEBwaCXVzLWVhc3QtMSJGMEQCIDPBzVlE8kKbBMiV9vG6ag678CIyhU7YlAHgVq8cmftgAiBkh8Z7U5bEJRxEb6O4JuhpsUC1QOAtImn8u7AQKgiwPyqpAgj1//////////8BEAIaDDUxMzkwNTcyMjc3NCIMroKIZCZIdAIzEalqKv0BUVfu40jguE//i7Tl64vJUTmJEt24jFhW99N/LjtEOpW72CJ/Mx1uvwLmL8qI1t+aQfNbtLDnXI9AGcaVosQH2jxmrNOzyWquuFmWCcqgaqYfpl61TNwaCfgz1pTWSZppEfS5SghquKM5vd8xNpQAsVuk8dhbRYOG0Y2CSyvkVAKU4G5dAciqgzSYRt7kqL8v8memXwDcfY9YTtMNkZjL6q/2GhswLt9yq+V0yZyPuZRCdS6cjoCEjzPoMxnVSiS99Ky7ePESoMxw6XFWOlsyOedCvpqPu7LMfu9fu23NXSlDG02DCixbyU+4LwjCR2LAOg44ZHw8i858PW9sWTCu0v2nBjqeATFAN3NO1HvwSCYFi1YX4VU2ivQplIe3o4mu0UA1WgWlx5XwIz6TGDkFoCmJd80sm5yIHOIn1L3xzQTZErAXJB+zJuD/7AZ2UEhKKtT7dglj2rrAz5FSGqvWOhYe1Ve5exLZgjDN9PtXiWWa/VgtCCf0TzKm2XX4G3S+hgqRxJQOoGdQ2QSHUpnSrQWrTg+QZkLA/Jnq+cvmmpV+ibmu'
        }
        # docker_conn_id='aws_default'
    )

    t1() >> t2


dag = docker_dag_multiple_commands()
