import torch






def create_model(name):
    if name == None:
        return None


    elif name == "Mamba++_ca2":
        from MambaPlusPlus_ca2 import Mamba_Plus_Plus_ca
        model = Mamba_Plus_Plus_ca()

    elif name == 'Mambair2':
        from mambairv2 import MambaIRv2SSR
        model = MambaIRv2SSR(
        img_size=64,
        in_chans=3,
        out_chans=31,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.).cuda()

        return model


    
    elif name == 'M3SR':
        from M3SR import M3SR
        model = M3SR(in_channels=3, out_channels=31).cuda()
        return model

    elif name in {'MFmamba', 'MFMamba'}:
        from MFMamba import MFmamba
        model = MFmamba()
        return model

    elif name in {'PIFNet', 'PIF-Net'}:
        from PIFNet import PIFNet
        model = PIFNet(hsi_bands=31, msi_bands=3)
        return model

    elif name in {'T_PMamba', 'T-PMamba', 'T_PMambaSR'}:
        from T_PMambaSR import T_PMambaSR
        model = T_PMambaSR(
            img_size=64,
            in_chans=3,
            out_chans=31,
            embed_dim=48,
            d_state=8,
            depths=[2, 2, 2, 2],
            num_heads=[4, 4, 4, 4],
            window_size=16,
            convffn_kernel_size=5,
            img_range=1.,
            mlp_ratio=1.)
        return model

    elif name == 'AWAN':
        from AWAN import AWAN
        model = AWAN()
        return model
    elif name == 'HSCNN':
        from HSCNN_Plus import HSCNN_Plus
        model = HSCNN_Plus()
        return model
    elif name == 'HSACS':
        from HSACS import HSACS
        model = HSACS()
        return model
    elif name == 'MST++':
        from MST_Plus_Plus import MST_Plus_Plus
        model = MST_Plus_Plus()
        return model
    elif name == 'HSRnet':
        from HSRnet import HSRnet
        model = HSRnet()
        return model
    elif name == 'LTRN':
        from LTRN import UDoubleTransformerCP3
        model = UDoubleTransformerCP3(3, 64, 64, 32, 31, 4, 1, "RTGB3", 3)
        return model

    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    return model
