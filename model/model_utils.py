from .cond_emb import ImgEncoder

from .denoiser import TransformerModel



def build_cond_encoder(config):
    if config.text_encoder == 'clip':
        module = ImgEncoder(config.img_encoder_path)
    else:
        raise ValueError("No such text encoder type!")

    return module


def build_denoiser(config):
    if config.denoiser == 'transformer':
        module = TransformerModel(act_dim=config.act_dim,
                                  con_dim=config.cond_dim,
                                  nhead=config.denoiser_nhead,
                                  hid_dim=config.hid_dim,
                                  nlayers=config.nlayers,
                                  dropout=config.dropout)
    else:
        raise ValueError("No such denoiser type!")
    return module