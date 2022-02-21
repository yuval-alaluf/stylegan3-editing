import torch
from torchvision import models

"""
Code is adopted from: AnyCostGAN (https://github.com/mit-han-lab/anycost-gan)
"""

URL_TEMPLATE = 'https://hanlab.mit.edu/projects/anycost-gan/files/{}_{}.pt'
attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
             'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
             'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def safe_load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False,
                                  file_name=None):
    # a safe version of torch.hub.load_state_dict_from_url in distributed environment
    # the main idea is to only download the file on worker 0
    try:
        import horovod.torch as hvd
        world_size = hvd.size()
    except:  # load horovod failed, just normal environment
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)

    if world_size == 1:
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
    else:  # world size > 1
        if hvd.rank() == 0:  # possible download... let it only run on worker 0 to prevent conflict
            _ = torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
        hvd.broadcast(torch.tensor(0), root_rank=0, name='dummy')
        return torch.hub.load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)


def load_state_dict_from_url(url, key=None):
    if url.startswith('http'):
        sd = safe_load_state_dict_from_url(url, map_location='cpu', progress=True)
    else:
        sd = torch.load(url, map_location='cpu')
    if key is not None:
        return sd[key]
    return sd


def get_pretrained(model, config=None):
    if model in ['attribute-predictor', 'inception']:
        assert config is None
        url = URL_TEMPLATE.format('attribute', 'predictor')  # not used for inception
    else:
        assert config is not None
        url = URL_TEMPLATE.format(model, config)

    if model == 'attribute-predictor':  # attribute predictor is general
        predictor = models.resnet50()
        predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
        predictor.load_state_dict(load_state_dict_from_url(url, 'state_dict'))
        return predictor
    else:
        raise NotImplementedError