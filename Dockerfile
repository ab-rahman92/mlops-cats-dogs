# Use official Python slim image (smaller than full python:3.11)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first → better caching for rebuilds
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code + saved model
COPY src/ ./src/
# Copy models folder only if it exists (prevents CI failure)
COPY --chmod=644 models/ ./models/  || true

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app with uvicorn (production-style, no --reload)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]