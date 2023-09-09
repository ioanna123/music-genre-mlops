from datetime import datetime

from airflow.decorators import task, dag
from airflow.providers.docker.operators.docker import DockerOperator


@dag(start_date=datetime(2021, 1, 1), schedule_interval='@daily', catchup=False)
def docker_dag():
    @task
    def t1():
        pass

    t2 = DockerOperator(
        task_id='t2',
        image='music-genre-classif:latest',
        command='python3 __main__.py train-using-image-features --model resnet18 --criterion cross_entropy --optimizer sdg --checkpoints_path checkpoints --images_path Data/images_original',
        network_mode='bridge',
        environment={
            'AWS_DEFAULT_REGION': 'us-east-1',
            'AWS_REGION': 'us-east-1',
            'AWS_SECRET_ACCESS_KEY': 's1ckVDxTGiH427mNmszYpNW7Gis5d5nOl9Rfe8TD',
            'AWS_ACCESS_KEY_ID': 'ASIAXPJZLAWLDSHOZ5EW',
            'AWS_SESSION_TOKEN': 'IQoJb3JpZ2luX2VjEAQaCXVzLWVhc3QtMSJGMEQCIBKAlnpx2bbpzMfkmQZv7PUschz2wTJdUqSeqETe3LxHAiAUhoWCEL+ySfCouarLljifWDw5JB7f9mnB/FMTt8TRWiqpAgjc//////////8BEAIaDDUxMzkwNTcyMjc3NCIMHEzBGnCN65htG5AWKv0B89H2PKbdAomuKcyV4gLICGLvKy5MdCxlWMi2vkUmQBI9R7s/BuPgkrtii5kHShC/DQCOr6y/10zVpREaEFPfmPRa89EFf8CXPU/g0hTbTFNijkC2FsD1I9DiVKvzAoNids68iY51bLX56SOUWk3HdTHPSWHRn3B3DBXzGxxbTXm/WxEBs3NYpoZzWKTWiSfCNw/Yj4AaD7NFroBYW/cxNxa8R0DWMkr/o3nkpXekLH4A3zTK5LAoe1zEXa8jl2LNKjm+rBPgxBaYgOMcE40GRKhjv9R/yblpbyiRRgXVOU4u+T7PJ0ojJvupsX8CmcIMpH5UCOXu3LA7G8rX8TDJrfinBjqeAfkTY/cfaK+ntBLA+5Y2yQTc73FBwVKOw3qp9Ja6Kll1EqeMN/qfR3kI+7r8JPek2rFO8VnrhaqFTqR0JWxYGDPH4AWJRwNAZhQXIXsyGOjb2YK4vcQ62+WvjtBb9rhR1JKo4FQB0VqYlDFniCepcmJsSpbw80o7cL2+71YCop0tZQNuamEVAACL64w6dWNisXAB9ti35H00RqKUHKNv'

        }
    )
    t1() >> t2


dag = docker_dag()
