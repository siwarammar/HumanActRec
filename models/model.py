from __future__ import division
import torch
from torch import nn
from models import resnext
import pdb
from .swin import SwinTransformer3D

def generate_model_resnext( opt):
    assert opt.model in ['resnext']
    assert opt.model_depth in [101]

    from models.resnext import get_fine_tuning_parameters
    model = resnext.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            cardinality=opt.resnext_cardinality,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            input_channels=opt.input_channels,
            output_layers=opt.output_layers)
    

    model = model.cuda()
    model = nn.DataParallel(model)
    
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        
        assert opt.arch == pretrain['arch']
        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = nn.Linear(model.module.fc.in_features, opt.n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()

def generate_model( opt):
    assert opt.model in ['swin-t','swin-s', 'swin-b', 'swin-sth']
    if opt.model=='swin-t':
        net = SwinTransformer3D(depths=[2, 2, 6, 2] , num_classes=400)
        linear_layer = nn.Linear(3072, opt.n_classes).cuda() #3072 #6144
    elif opt.model =='swin-s':
        net = SwinTransformer3D(depths=[2, 2, 18, 2], num_classes=400)
        linear_layer = nn.Linear(24576, opt.n_classes).cuda()
        

    elif opt.model =='swin-b':
        net =  SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], num_classes=400)
        linear_layer = nn.Linear(24576, opt.n_classes).cuda()   #32768 #8192 #16384
    elif opt.model == 'swin-sth':
        net =  SwinTransformer3D(patch_size=(2,4,4), window_size=(16,7,7), drop_path_rate=0.4, depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32], num_classes=174)
        linear_layer = nn.Linear(8192, opt.n_classes).cuda()

    if opt.pretrain_path!=None:
        ckpt = torch.load(opt.pretrain_path)
        ckpt_dict = load_weights(ckpt['state_dict'])
        net.load_state_dict(ckpt_dict,strict=False)
        print("loaded weights in the model")
    net = net.cuda()
    net = nn.DataParallel(net)
    linear_layer = nn.DataParallel(linear_layer)   
    model = nn.Sequential(net, linear_layer)

    return model, model.parameters()




def load_weights(state_dict):
    state_dict_new = {}
    for key, value in state_dict.items():
        if "backbone." in key:
            state_dict_new[key.replace("backbone.","")]=value
        elif 'cls_head.' in key:
            state_dict_new[key.replace("cls_head.","")]=value
        else:
            state_dict_new[key]=value
    return state_dict_new
