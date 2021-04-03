FROM python:3.6-slim

RUN apt-get update

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . .

RUN mkdir ./Server/Model
ADD https://doc-0c-38-docs.googleusercontent.com/docs/securesc/gnd6cdveajdu52kk4qfrag4g9b86pndl/372tl43jm9r038na66ad5eva5595levg/1617432225000/14069255002135782085/10786148373978228622Z/1-1sx4qC08MgE7XZwd0MwFQpvYtMrXxGZ?e=download&nonce=c241ivnj1mk0s&user=10786148373978228622Z&hash=on06d3uis96a2j16ju77o70u8acbgblk ./Server/Model
ADD https://doc-0s-38-docs.googleusercontent.com/docs/securesc/gnd6cdveajdu52kk4qfrag4g9b86pndl/fsesghnnu4gvj6leqk5fhb4mhn8go8in/1617432600000/14069255002135782085/10786148373978228622Z/1Lk3dflSpm_c0vUs88tY4GjETYqpI8Gfb?e=download&nonce=rpo1adev5das4&user=10786148373978228622Z&hash=hvvgjg3u8jvs5tj46lblhklcp6ipts34 ./Server/Model

RUN mkdir ./Server/Token
ADD https://doc-0k-38-docs.googleusercontent.com/docs/securesc/gnd6cdveajdu52kk4qfrag4g9b86pndl/8fg7j52pvhfcqhmcvn3a8d9gd8cr4943/1617432750000/14069255002135782085/10786148373978228622Z/1-3p_g-iuMG1brqZxEHPL2eb6Sk1vaDpS?e=download ./Server/Token
ADD https://doc-0s-38-docs.googleusercontent.com/docs/securesc/gnd6cdveajdu52kk4qfrag4g9b86pndl/jok27pjd7pqooe0so6vgcngv8ucnm00f/1617432750000/14069255002135782085/10786148373978228622Z/1-16h1St1lyvvZKyot-jggeDtXVn3RarT?e=download ./Server/Token
ADD https://doc-10-38-docs.googleusercontent.com/docs/securesc/gnd6cdveajdu52kk4qfrag4g9b86pndl/vi2m3kgn2t2s54l8sija5jmsndceucd9/1617432825000/14069255002135782085/10786148373978228622Z/1-1XAA6TSFQO5a1SWattirsoIsMtBG4oO?e=download ./Server/Token

EXPOSE 5000

EXPOSE $PORT

WORKDIR ./Server

CMD ["python", "server.py"]
