import timm

from timm.models.vision_transformer import VisionTransformer
from torch.nn import LayerNorm
from models.wideresnet import WideResNet
from models.unet import ContextUnet

def create_model(model_name, num_classes, input_size=32):
    match model_name:
        case 'context-unet':
            model = ContextUnet(3, input_size, input_size, 16, num_classes, 1)
        case 'wrn-40-4':
            model = WideResNet(40, num_classes, 4)
        case 'wrn-16-4':
            model = WideResNet(16, num_classes, 4)
        case 'vit-tiny':
            model = VisionTransformer(
                img_size=input_size, 
                patch_size=4,
                embed_dim=192,
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                num_classes=num_classes,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case 'vit-small':
            model = VisionTransformer(
                img_size=input_size, 
                patch_size=4,
                embed_dim=384,
                depth=12, 
                num_heads=6, 
                mlp_ratio=4,
                num_classes=num_classes,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case 'vit-base':
            model = VisionTransformer(
                img_size=input_size, 
                patch_size=4,
                embed_dim=768,
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                num_classes=num_classes,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case 'vit-large':
            model = VisionTransformer(
                img_size=input_size, 
                patch_size=4,
                embed_dim=1024,
                depth=24, 
                num_heads=16, 
                mlp_ratio=4,
                num_classes=num_classes,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case _:
            model = timm.create_model(model_name, num_classes=10)

    return model
