FROM gcc:latest

RUN apt-get update && \
    apt-get install -y \

RUN RUN  git clone  --branch=master https://github.com/DenisReznikov/YOLOv4_R.git


# Выполним сборку нашего проекта, а также его тестирование
RUN ls 

