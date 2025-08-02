FROM python:3.10.0-slim-buster
WORKDIR /DisCoveR-py
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt
COPY . .
ADD https://data.4tu.nl/ndownloader/files/24073733 data/Hospital_log.xes.gz
ADD https://data.4tu.nl/ndownloader/files/24027287 data/BPI_Challenge_2012.xes.gz
ADD https://data.4tu.nl/ndownloader/files/24063818 data/BPIC15_1.xes
ADD https://data.4tu.nl/ndownloader/files/24018146 data/Road_Traffic_Fine_Management_Process.xes.gz
RUN gunzip data/*.gz
#RUN python main.py -i
#RUN python main.py -f -a
#RUN python main.py -f && python main.py -a
