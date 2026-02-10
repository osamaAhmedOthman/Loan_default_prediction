# 1. Use an official Python runtime as a parent image
FROM python:3.10-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY app/requirements.txt .

# 4. Install the needed packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code and the model
# This copies everything from your local 'app' folder to the container's '/app'
COPY app/ .

# 6. Make port 8000 available to the world outside this container
EXPOSE 8000

# 7. Run uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]