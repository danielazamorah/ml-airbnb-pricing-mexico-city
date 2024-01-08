FROM python:3.10-slim

WORKDIR /app

COPY ["streamlit_app.py", "requirements_st.txt", "./"]

RUN pip3 install -r requirements_st.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]