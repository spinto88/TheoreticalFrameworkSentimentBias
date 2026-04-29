#!/bin/bash
PID_FILE="server.pid"
LOG_FILE="server.log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
    echo "El servidor ya está corriendo (PID $(cat $PID_FILE))."
    exit 1
fi

nohup uvicorn src.app:app --host 0.0.0.0 --port 8000 >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "Servidor iniciado (PID $!, log: $LOG_FILE)"
echo "Para apagarlo: kill \$(cat $PID_FILE)"
