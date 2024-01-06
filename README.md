# My Summary App

Welcome to My Summary App! This application provides a summary of text input using advanced natural language processing techniques.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Contributing](#contributing)
- [License](#license)

## Introduction

My Summary App is a powerful tool for generating summaries from pdf files. It utilizes state-of-the-art natural language processing algorithms to extract key information and provide concise summaries.

## Installation

To install and run My Summary App locally, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the application using `streamlit run src/app/app.py`.
4. Open your web browser and navigate to `http://localhost:8501` to access the app.

## Usage

Once the application is running, you can drag and drop any pdf file (limit 200MB per file) or use the button 'Browse file' to obtain a summary of the input pdf text. The generated summary will be displayed on the screen.

## Docker

Alternatively, you can run My Summary App using Docker. Docker provides a lightweight and portable environment for running applications.

To run My Summary App using Docker, follow these steps:

1. Install Docker on your machine by following the instructions in the [official Docker documentation](https://docs.docker.com/get-docker/).
2. Pull the Docker image for My Summary App from Docker Hub using the following command:
    docker pull `<username>/<repository>:<tag>`
    Replace `<username>/<repository>:<tag>` with the appropriate values for the image you want to pull. For example:
    docker pull yousseffhm/text-summarization:latest
3. Run the Docker image using the following command:
    docker run -p 8501:8501 `<username>/<repository>:<tag>`
4. Open your web browser and navigate to `http://localhost:8501` to access the app.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).