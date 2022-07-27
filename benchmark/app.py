from crypt import methods
from unittest import result
from flask import Flask, redirect,render_template, request, session, url_for,flash
import sys
import os

import json
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path) 
from example.web_page import My_Model

app=Flask(__name__,template_folder='templates', static_url_path='/', static_folder='mega')


# 调用模型
re = My_Model(True)



class Save_data():
    def __init__(self,id=-1,token=[]):
        self.id = id 
        self.token = token
save_data = Save_data()


def change(ls,s,e,is_head=True):
    for k in range(s,e):
        if is_head:
            ls[k] = "<b class=\"head_entity\">"+ls[k]+"</b>"
        else:
            ls[k] = "<b class=\"tail_entity\">"+ls[k]+"</b>"



@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/info/<id>',methods=['get'])
def info(id):
    d=request.args.get('direction')
    id = int(id)
    if 0<= id < 12247:
        img_pos = "train"
    elif id < 13871:
        img_pos = "val"
        id -= 12247
    elif id < 15485:
        img_pos = "test"
        id -= 13871
    #re.find_my_file(0,[],[])
   
    
   
    data = re.get_item(id)
    
    save_data.id = id
    save_data.token = data["token"].copy()
    print("save",save_data.id)
    print("save",save_data.token)
    data["img_id"] = img_pos+"/"+data["img_id"]
    data["sentence"] = " ".join(data["token"])
    data["id"] = id 
    pos_h = data["h"]["pos"]
    pos_t = data["t"]["pos"]
    
    change(data["token"],pos_h[0],pos_h[1])
    change(data["token"],pos_t[0],pos_t[1],False)
    data["sentence_html"] = " ".join(data["token"])
    data["token_html"]=data["token"]
    #print("this is id ",id)
    #print(data)
    # data={'datas':{'h':{'name':name,'pos':pos},'img_id':img_id,'img_path':img_path,"relation":relation,"sentence":sentence,"sentence_html":sentence_html,"t":{"name":name_t,"pos":pos_t},"token":token,"token_html":token_html},"id":id}
    return json.dumps(data)





@app.route('/run',methods=["post"])
def run():
    data = request.json
    pos_h,pos_t = [],[]
    get_head,get_tail = [],[]
    if len(data["head"])==0:
        pos_h = []
    else:
        get_head = data["head"].split()
    if len(data["tail"])==0:
        pos_t = []
    else:
        get_tail = data["tail"].split()
    print(save_data.id)
    print("my token is",save_data.token)
    try:
        if len(get_head) != 0:
            pos_h = [save_data.token.index(get_head[0]),save_data.token.index(get_head[-1])+1]
        if len(get_tail) != 0:
            pos_t = [save_data.token.index(get_tail[0]),save_data.token.index(get_tail[-1])+1]
        re.find_my_file(save_data.id,pos_h,pos_t)
        result = re.eval()
        print(result)
        return json.dumps({"result":result})
    except ValueError:
        return json.dumps({"result":"the entity not in the sentence"})
    except:
        return json.dumps({"result":"another cause"})
    

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
