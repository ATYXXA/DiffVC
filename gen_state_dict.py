import params
import torch
from model.vc import ControlledDiffVC, ControlledDiffVC_emo


##############################
n_mels = params.n_mels
channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim
dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = False
beta_min = params.beta_min
beta_max = params.beta_max
##############################


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


if __name__ == "__main__":
    model = ControlledDiffVC_emo(n_mels, channels, filters, heads, layers, kernel, 
                dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                dec_dim, beta_min, beta_max).cuda()
    ##############################
    ckpt = 'pretrained_vc43.pt'           
    pretrained_weights = torch.load(ckpt)
    ##############################
    scratch_dict=model.state_dict()
    target_dict={}
    for key in scratch_dict.keys():

        is_control, name = get_node_name(key, 'decoder.controller')
        if is_control:
            # print(key)
            copy_key = 'decoder.estimator' + name
        else:
            copy_key = key

        if copy_key in pretrained_weights:
            target_dict[key] = pretrained_weights[copy_key].clone()
        else:
            target_dict[key] = scratch_dict[key].clone()
            print("Newly added: ", key)

    model.load_state_dict(target_dict,strict=True)
    torch.save(model.state_dict(), f'emo_from_vc43.pt')
    print('Done!')
