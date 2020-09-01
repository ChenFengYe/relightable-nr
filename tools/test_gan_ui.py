from flask import Flask
from flask import request, jsonify
from flask import send_file
from io import BytesIO

from PIL import Image

from test_gan import build_model
import numpy as np
import json

import torch
import cv2
import math

class Args():
    def __init__(self, cfg, opts):
        self.cfg = cfg
        self.opts = opts

args = Args('/new_disk/chenxin/relightable-nr/data/200830_hnrd_SDAP_30714418105/logs/08-31_07-21-59_NoAtt_linear/200830_GAN_APose.yaml',
            ['WORKERS','0', 'TEST.BATCH_SIZE','1'])


app = Flask(__name__)

def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# inference loading for ui
def prepare_camera_transform(obj, Ts):
    if Ts is not None:
        Ts = np.linalg.inv(Ts)
        Ts = torch.from_numpy(Ts.astype(np.float32))

    in_points = obj['v_attr']['v']

    center = torch.mean(in_points,dim=0).cpu()
    # up = -torch.mean(Ts[:,0:3,0],dim =0)
    up = -torch.tensor([0.,0.,1.])
    # up = -Ts[0:3,0]
    up = up / torch.norm(up)
        
    num_points = torch.Tensor([in_points.size(0)])
    
    in_points = in_points.cuda()

    radius = torch.norm(Ts[0:3,3] - center) *1.0
    # center = center + up*radius*0.35
    center = center

    v = torch.randn(3)
    v = v - up.dot(v)*up
    v = v / torch.norm(v)

    s_pos = center + v * radius

    # s_pos = Ts[0,0:3,3]
    center = center.numpy()
    up = up.numpy()

    radius = radius.item()
    s_pos = s_pos.numpy()

    global pos
    global xaxis
    pos = s_pos
    lookat = center - pos
    dis=np.linalg.norm(lookat)/100
    lookat = lookat/np.linalg.norm(lookat)
    xaxis = np.cross(lookat, up)
    xaxis = xaxis / np.linalg.norm(xaxis)

    cam_data = {}
    cam_data['center'] = center
    cam_data['up'] = up
    cam_data['dis'] = dis
    return cam_data

def calculate_cam_pose(data, cam_data):
    center = cam_data['center']
    up = cam_data['up']
    dis = cam_data['dis']
    
    # calculate cam
    op=data['op']        
    angle = 3.1415926*2/360.0    
    
    global pos
    global xaxis
    pos = pos - center
    if op[0]==1:
        print('LeftLook')
        pos = rodrigues_rotation_matrix(up,-angle).dot(pos) 
    elif op[0]==2:
        print('RightLook')
        pos = rodrigues_rotation_matrix(up,angle).dot(pos) 
    elif op[0]==3:
        print('UpLook')
        pos = rodrigues_rotation_matrix(xaxis,-angle).dot(pos) 
    elif op[0]==4:
        print('DownLook')
        pos = rodrigues_rotation_matrix(xaxis,angle).dot(pos) 
    elif op[0]==5:
        print('ZoomIn')
        pos = pos-dis*pos/np.linalg.norm(pos)
    elif op[0]==6:
        print('ZoomOut')
        pos = pos+dis*pos/np.linalg.norm(pos)
    pos = pos + center
        
    lookat = center - pos
    lookat = lookat/np.linalg.norm(lookat)
    
    # yaxis = -np.cross(lookat, up)
    # yaxis = yaxis / np.linalg.norm(yaxis)
        
    # xaxis = np.cross(yaxis,lookat)
    # xaxis = xaxis/np.linalg.norm(xaxis)
    
    xaxis = np.cross(lookat, up)
    xaxis = xaxis / np.linalg.norm(xaxis)
    
    yaxis = -np.cross(xaxis,lookat)
    yaxis = yaxis/np.linalg.norm(yaxis)
    
    nR = np.array([xaxis,yaxis,lookat, pos]).T
    nR = np.concatenate([nR,np.array([[0,0,0,1]])])
    
    T = torch.Tensor(nR)

    # world2cam
    T = np.linalg.inv(T.numpy())
    T = torch.Tensor(T).cuda()
    return T

# load model
model, view_dataloader, view_dataset, save_dict = build_model(args)

# prepare cam
obj = view_dataset.objs[0]
Ts = view_dataset.poses_all[0]
Ts = np.dot(Ts, view_dataset.global_RT_inv)
cam_data = prepare_camera_transform(obj, Ts)
global view_data
view_data = next(iter(view_dataloader))    

@app.route('/', methods = ["GET","POST"])
def hello_world():
    # recevice data
    data = request.get_data()
    data = json.loads(data)

    # generate view
    # view_data = view_dataset.__getitem__(0)
    T = calculate_cam_pose(data, cam_data)
    global view_data
    # T = view_data['pose'][0,...]
    view_data = view_dataset.read_view_from_cam(view_data, T)

    # inference
    model.set_input(view_data)
    model.test(view_data)
    outputs = model.get_current_results()

    outputs_img = outputs['img_rs']
    neural_img = outputs['nimg_rs']
    # uv_map = view_data['uv_map']

    outputs_img = outputs_img.detach().cpu()[0]
    outputs_img = cv2.cvtColor(outputs_img.permute(1,2,0).numpy()*255.0, cv2.COLOR_BGR2RGB)

    neural_img = neural_img.detach().cpu()[0]
    neural_img = cv2.cvtColor(neural_img.permute(1,2,0).numpy()*255.0, cv2.COLOR_BGR2RGB)
    # Im = Image.fromarray(outputs_img.astype('uint8')).convert('RGB')

    # mask = mask_t.permute(1,2,0).numpy()*255.0
    # rgba=img*mask/255.0+(255.0-mask)*bg/255.0

    # Im = Image.open("/new_disk/chenxin/relightable-nr/data/200830_hnrd_SDAP_30714418105/img/SDAP_30714418105_80_00000.jpg")
    # print(Im)

    # mask = np.concatenate([mask,mask,mask],axis=2)
    # depth_t = depth.detach().cpu()[0]
    # depth_res = depth_t.permute(1,2,0).numpy()*255.0
    # depth_res = np.concatenate([depth_res,depth_res,depth_res],axis=2)

    Im_res = np.hstack((outputs_img, neural_img))
    Im = Image.fromarray(Im_res.astype('uint8')).convert('RGB')
    return serve_pil_image(Im)

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0',port=800)