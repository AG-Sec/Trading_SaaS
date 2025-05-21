# Trading SaaS Platform

This project is a Software as a Service (SaaS) platform for algorithmic trading.

## Project Overview

(As per PRD)

## Tech Stack

- **Backend**: Python 3.12.10, FastAPI, Pydantic
- **Frontend**: Next.js (App Router), React, TypeScript, Tailwind CSS, shadcn/ui
- **Data Analysis**: Pandas
- **Market Data**: yfinance (initially), with adapters for others
- **Technical Analysis**: TA-Lib
- **Database**: (To be decided - PostgreSQL, SQLite, etc.)
- **Deployment**: Docker, (Cloud provider TBD)

## Project Structure

- `/backend`: Contains the FastAPI application, core agent logic, API endpoints.
- `/frontend`: Contains the Next.js application, UI components, and user dashboard.
- `/shared_types`: Contains Pydantic models or TypeScript types shared between backend and frontend.
- `/docs`: Project documentation, including PRD, ADRs, etc.
- `docker-compose.yml`: For local development and services orchestration.
- `.env.template`: Template for environment variables.

## Getting Started

### Prerequisites

- Python 3.12.10 (managed with `pyenv` is recommended)
- Node.js and npm (Next.js project was initialized with npm)
- Docker and Docker Compose
- `pyenv` (recommended for Python version management, install via `curl https://pyenv.run | bash` or see official docs)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Trading_SaaS
    ```

2.  **Set up Python environment (using pyenv):**
    Ensure you have pyenv installed. The `.python-version` file in this repository specifies `3.12.10`.
    ```bash
    # pyenv should automatically pick up the version from .python-version if you cd into the directory
    # If not, or for the first time:
    pyenv install $(cat .python-version) # Or pyenv install 3.12.10
    pyenv virtualenv $(cat .python-version) trading_saas
    pyenv local trading_saas # This creates/updates .python-version if not present
    # Activate the virtual environment (though pyenv local often handles this for the current shell session)
    # source $(pyenv root)/versions/$(cat .python-version)/envs/trading_saas/bin/activate
    pip install -r backend/requirements.txt
    ```

3.  **Set up frontend dependencies:**
    The frontend project was initialized using `create-next-app` with `npm`.
    ```bash
    cd frontend
    npm install
    cd ..
    ```

4.  **Configure environment variables:**
    Copy `.env.template` to `.env` and fill in the required values for both backend and frontend.
    ```bash
    cp .env.template .env
    ```
    *Ensure `NEXT_PUBLIC_API_URL` in `.env` points to your backend (e.g., `http://localhost:8000` or `http://backend:8000` if running via Docker Compose and frontend needs to call backend service name).*

5.  **Build and Run with Docker Compose:**
    This is the recommended way to run the application for a production-like setup or for integrated testing.
    Ensure Dockerfiles exist in `./backend/Dockerfile` and `./frontend/Dockerfile`.
    ```bash
    docker-compose up -d --build
    ```
    - Backend will be available at [http://localhost:8000](http://localhost:8000)
    - Frontend will be available at [http://localhost:3000](http://localhost:3000)

## Development

For a more iterative development workflow, you can run the backend and frontend services separately.

### Backend (FastAPI)

```bash
cd backend
# Ensure your pyenv virtual environment 'trading_saas' is active
# pyenv activate trading_saas  (if not already)
# or source $(pyenv root)/versions/$(cat ../.python-version)/envs/trading_saas/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port ${FASTAPI_PORT:-8000} # Uses FASTAPI_PORT from .env or defaults to 8000
```

### Frontend (Next.js)

```bash
cd frontend
npm run dev
```
Frontend will be accessible at [http://localhost:3000](http://localhost:3000).

## Milestones

(As per PRD and GitHub Issues)

## Contributing

(Contribution guidelines TBD)

## License

(License TBD)
