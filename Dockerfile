FROM python:3.8
RUN apt update && apt install -y python3-pip
RUN pip3 install matplotlib torch torchvision Flask
#RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu
#RUN pip3 install torch torchvision -f https://download.pytorch.org/whl/cu<version>/torch_stable.html
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000
COPY app.py app/
COPY model.py app/
COPY utils.py app/
COPY dataset/ app/
COPY last_chkp.pth app/
CMD ["python3", "app/app.py" ]