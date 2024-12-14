

from flask import Flask, render_template, request, jsonify
from Model.t5 import getT5Model
from Model.bart import summarize,summarize_tuned

app = Flask(__name__)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# API 路由：处理输入并返回结果



@app.route('/submit', methods=['POST'])
def submit():
    input = request.form['input']  # 获取表单数据
    model=request.form['model']
    if model=="BART":
        result=summarize(input)
    elif model=="BARTTUNED":
        result=summarize_tuned(input)
    else:
        result=("请输入正确的模型")
    # 在这里可以添加处理逻辑，比如存储数据、进行计算等
    return render_template('index.html', result=result,input=input,model=model)
    

if __name__ == '__main__':
    app.run(debug=True)