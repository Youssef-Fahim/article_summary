# Use a base image with Python and Streamlit pre-installed
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /myapp

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files to the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Add metadata to the image to describe which port the container is listening on at runtime.
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the command to run the Streamlit app
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# docker run command: docker run -p 8501:8501 my-summary-app:v1.0.2