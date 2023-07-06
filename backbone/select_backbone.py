from .s3dg import S3D
from .resnet_2d3d import r2d3d50
import sys
sys.path.append("..")
from modeling_finetune import vit_base_patch16_224, vit_base_patch16_112

def select_backbone(network, first_channel=3):
# def select_backbone(network, first_channel=3, img_size=224):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r50':
        param['feature_size'] = 2048
        model = r2d3d50(input_channel=first_channel)
    elif network == 'vit':
        param['feature_size'] = 768
        # if img_size == 224:
        #     model = vit_base_patch16_224()
        # elif img_size == 112:
        #     model = vit_base_patch16_112()
        # model = vit_base_patch16_224()
        model = vit_base_patch16_112()
    else: 
        raise NotImplementedError

    return model, param

from torchsummary import summary
# model = S3D(input_channel=3)
# summary(model.cuda(), input_size=(3, 16, 224, 224))
# model = vit_base_patch16_224()
# summary(model.cuda(), input_size=(3, 16, 224, 224))