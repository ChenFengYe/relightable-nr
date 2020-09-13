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
import time

class Args():
    def __init__(self, cfg, opts):
        self.cfg = cfg
        self.opts = opts

# fashion video
model_path = '/home/chenxin/relightable-nr/data/200909_fashion_small/logs/09-10_03-35-40_ganhd_mask/200909_GANHD_Contextual_mask.yaml'
# model_path = '/home/chenxin/relightable-nr/data/200909_fashion_small/logs/09-10_06-18-35_ganhd_mask/200909_GANHD_Contextual_mask.yaml'

# trump
# model_path = '/new_disk/chenxin/relightable-nr/data/200906_trump/logs/09-06_11-04-21_test_8_trump_from_internet/200903_GAN_APose.yaml'

# # sport short male running
# model_path = '/new_disk/chenxin/relightable-nr/data/200903_justin/logs/09-03_16-02-04_cam_9views/200903_GAN_APose.yaml' 

# # shirt female Apose
# model_path = '/new_disk/chenxin/relightable-nr/data/200830_hnrd_SDAP_14442478293/logs/09-02_16-34-18_cam_9views/200830_GAN_APose.yaml' 

args = Args(model_path, ['WORKERS','0', 'TEST.BATCH_SIZE','1'])

# args = Args('/new_disk/chenxin/relightable-nr/data/200830_hnrd_SDAP_30714418105/logs/08-31_07-21-59_NoAtt_linear/200830_GAN_APose.yaml',
#             ['WORKERS','0', 'TEST.BATCH_SIZE','1'])

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
    up = torch.tensor([0.,1.,0.])
    # up = -torch.tensor([0.,0.,1.])
    # up = torch.tensor([0.,0.,1.])
    # up = torch.tensor([0.,1.,0.]) # dome camera
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

    global global_pos
    global xaxis
    global_pos = s_pos
    lookat = center - global_pos
    dis=np.linalg.norm(lookat)/100
    lookat = lookat/np.linalg.norm(lookat)
    xaxis = np.cross(lookat, up)
    xaxis = xaxis / np.linalg.norm(xaxis)

    cam_data = {}
    global global_center
    global_center = center
    cam_data['center'] = global_center
    cam_data['up'] = up
    cam_data['dis'] = dis
    return cam_data

def control_cam(data):
    op=data['op']    
    global control_speed
    global is_move
    global is_rotate
    if op[0] == 9:
        is_rotate = not is_rotate
    elif op[0] == 10:
        control_speed = control_speed+1
    elif op[0] == 11:
        control_speed = control_speed-1
    elif op[0] == 12:
        tmp = is_move 
        is_move = not tmp

    if is_rotate:
        op[0] = 1 
    return data

def calculate_cam_pose(data, cam_data):
    global control_speed
    global global_pos
    global global_center
    global xaxis

    # global_center = cam_data['center']
    up = cam_data['up']
    dis = cam_data['dis']*2**control_speed
    
    # calculate cam
    op=data['op']
    angle = 3.1415926*2/360.0*(2**control_speed)
    
    global_pos = global_pos - global_center

    global is_move
    if not is_move:
        if op[0]==1:
            print('LeftLook')
            global_pos = rodrigues_rotation_matrix(up,-angle).dot(global_pos) 
        elif op[0]==2:
            print('RightLook')
            global_pos = rodrigues_rotation_matrix(up,angle).dot(global_pos) 
        elif op[0]==3:
            print('UpLook')
            global_pos = rodrigues_rotation_matrix(xaxis,-angle).dot(global_pos) 
        elif op[0]==4:
            print('DownLook')
            global_pos = rodrigues_rotation_matrix(xaxis,angle).dot(global_pos) 
    else:
        move_step = 0.05
        if op[0]==1:
            print('LeftLook')
            global_center = global_center + move_step*xaxis
        elif op[0]==2:
            print('RightLook')
            global_center = global_center - move_step*xaxis
        elif op[0]==3:
            print('UpLook')
            global_center = global_center - move_step*up
        elif op[0]==4:
            print('DownLook')
            global_center = global_center + move_step*up

    if op[0]==5:
        print('ZoomIn')
        global_pos = global_pos-dis*global_pos/np.linalg.norm(global_pos)
    elif op[0]==6:
        print('ZoomOut')
        global_pos = global_pos+dis*global_pos/np.linalg.norm(global_pos)
    global_pos = global_pos + global_center
        
    lookat = global_center - global_pos
    lookat = lookat/np.linalg.norm(lookat)
    
    # yaxis = -np.cross(lookat, up)
    # yaxis = yaxis / np.linalg.norm(yaxis)
        
    # xaxis = np.cross(yaxis,lookat)
    # xaxis = xaxis/np.linalg.norm(xaxis)
    
    xaxis = np.cross(lookat, up)
    xaxis = xaxis / np.linalg.norm(xaxis)
    
    yaxis = -np.cross(xaxis,lookat)
    yaxis = yaxis/np.linalg.norm(yaxis)
    
    nR = np.array([xaxis,yaxis,lookat, global_pos]).T
    nR = np.concatenate([nR,np.array([[0,0,0,1]])])
    
    T = torch.Tensor(nR)

    # world2cam
    T = np.linalg.inv(T.numpy())
    T = torch.Tensor(T).cuda()
    return T

def calculate_frame(data, frame_id, frame_range):
    frame_id = frame_id.item()
    op=data['op']
    frame_len = len(frame_range)
    idx = frame_range.index(frame_id)
    if op[0]==7:
        idx = (idx-1)%frame_len
        print('previous frame')
    elif op[0]==8:
        idx = (idx+1)%frame_len
        print('previous frame')
    cur_frame_id = frame_range[idx]
    return torch.tensor([cur_frame_id])

# load model
model, view_dataloader, view_dataset, save_dict = build_model(args)

# prepare cam
global global_view_data
global_view_data = next(iter(view_dataloader))    
obj = view_dataset.objs[global_view_data['f_idx'].item()]
Ts = view_dataset.poses_all[0]
Ts = np.dot(Ts, view_dataset.global_RT_inv)
cam_data = prepare_camera_transform(obj, Ts)

# prepare interaction
global is_move
is_move = False
global is_rotate
is_rotate = False
global control_speed
control_speed = 0.0

global rotate_count
rotate_count = 0
@app.route('/', methods = ["GET","POST"])
def hello_world():
    t_start_all = time.time()
    t_start = time.time()

    # recevice data
    data = request.get_data()
    data = json.loads(data)

    # generate view
    # view_data = view_dataset.__getitem__(0)
    # T = view_data['pose'][0,...]
    data = control_cam(data)
    T = calculate_cam_pose(data, cam_data)

    global global_view_data
    global_view_data['f_idx'] = calculate_frame(data, global_view_data['f_idx'], view_dataset.frame_range)    
    global_view_data = view_dataset.read_view_from_cam(global_view_data, T)

    # build calib
    global is_rotate
    global rotate_count
    if is_rotate:
        view_dataset.calib['poses'][rotate_count, ...] = global_view_data['pose'][0, ...].clone().detach().cpu().numpy()
        view_dataset.calib['projs'][rotate_count, ...] = global_view_data['proj'][0, ...].clone().detach().cpu().numpy()
        view_dataset.calib['dist_coeffs'][rotate_count, ...] = global_view_data['dist_coeff'][0, ...].clone().detach().cpu().numpy()
        rotate_count += 1
        if rotate_count == 360:
            import scipy
            scipy.io.savemat('/home/chenxin/relightable-nr/data/200909_fashion_small/calib/calib_0911_rendered360_fix2.mat', view_dataset.calib)


    # inference
    model.set_input(global_view_data)

    print('load data'+ str(time.time()-t_start) + '  s')
    t_start = time.time()

    model.test(global_view_data)
    print('test data'+ str(time.time()-t_start) + '  s')
    t_start = time.time()

    outputs = model.get_current_results()

    outputs_img = outputs['rs'][:,0:3,:,:]
    neural_img = outputs['nimg_rs']
    # uv_map = global_view_data['uv_map']

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
    im_data = serve_pil_image(Im)

    print('outp data'+ str(time.time()-t_start) + '  s')
    print('all  time'+ str(time.time()-t_start_all) + '  s')
    return im_data

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0',port=8030)
    