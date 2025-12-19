# api/index.py
from app import app

# Vercel规定：变量名必须叫"application"，用来接收app.py中的Flask实例
application = app

