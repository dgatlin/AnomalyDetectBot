ARG REGION=us-east-1

FROM XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3

ENV PATH="/opt/ml/code:${PATH}"

COPY /code /opt/ml/code
COPY anom_model/dist/anom_model-0.0.1-py3-none-any.whl /opt/anom_model-0.0.1-py3-none-any.whl

RUN pip install -r /opt/ml/code/requirements.txt /opt/gam_model-0.0.1-py3-none-any.whl

ENV SAGEMAKER_PROGRAM model_logic.py
