#!/bin/bash
#uvicorn src.app:app --reload
uvicorn src.app:app --host 0.0.0.0 --port 8000
