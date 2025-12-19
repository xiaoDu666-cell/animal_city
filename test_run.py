# 测试Flask是否正常工作
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask安装成功！可以正常使用了。"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
