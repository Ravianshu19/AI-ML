FROM pytorch/torchserve:latest-cpu

USER root

COPY serving/handler.py /home/model-server/handler.py
COPY model/fraud_model.pt /home/model-server/fraud_model.pt
COPY serving/config.properties /home/model-server/config.properties

WORKDIR /home/model-server

RUN torch-model-archiver \
    --model-name fraud \
    --version 1.0 \
    --serialized-file fraud_model.pt \
    --handler handler.py \
    --export-path /home/model-server/model-store \
    --force

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--ts-config", "config.properties", \
     "--model-store", "/home/model-server/model-store", \
     "--models", "fraud=fraud.mar", "--foreground"]
