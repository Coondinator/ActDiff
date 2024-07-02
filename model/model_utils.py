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
        module = TransformerModel(d_model=config.latent_size,
                                  nhead=config.denoiser_nhead,
                                  d_hid=config.latent_size,
                                  nlayers=config.denoiser_nlayer)
    else:
        raise ValueError("No such denoiser type!")
    return module