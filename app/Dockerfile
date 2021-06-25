# Base image of python 3.7
FROM python:3.7

# Setup the working directory
WORKDIR /app

# Copy files from the current directory to the working directory
COPY . /app

# Install all the dependencies for the app
RUN pip install -r requirements.txt

# Expose port 8080 of the container for listening
EXPOSE 8080

# The command to launch app and expose it's port 8080, when container is up and running
CMD python main.py --server.port=8080
