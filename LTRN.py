import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np

class MSA(nn.Module):
    def __init__(self, num_vector, num_heads_column, heads_number):
        super(MSA, self).__init__()
        self.num_vector = num_vector
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(num_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))
        self.proj = nn.Linear(num_heads_column * heads_number, num_vector)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(num_heads_column * heads_number, num_heads_column * heads_number,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.modules.activation.GELU(),
            nn.Conv2d(num_heads_column * heads_number, num_heads_column * heads_number,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, x_in):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))
        v = v
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.heads_number * self.num_heads_column)
        out_c = self.proj(x).view(b, h, w, c)

        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return [out_c, out_p]


class ResNetBlock2(nn.Module):
    def __init__(self, c_in):
        super(ResNetBlock2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=(1, 1), bias=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c_in // 2, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c_in // 2, c_in // 2, kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(c_in // 2, c_in, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + x
        out = F.leaky_relu(out, negative_slope=0.2, inplace=False)
        return out


class RTGB1(nn.Module):
    def __init__(self, num_direction):
        super(RTGB1, self).__init__()
        self.rtgb_average = nn.AdaptiveAvgPool2d((1, 1))
        self.rtgb_max = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_share = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_average = self.rtgb_average(x)
        x_max = self.rtgb_max(x)
        x_all = self.conv_share(x_average) + self.conv_share(x_max)
        out = self.conv(x_all)
        return out


class RTGB2(nn.Module):
    def __init__(self, num_direction, width, height):
        super(RTGB2, self).__init__()
        self.rtgb_depthwise = nn.Conv2d(num_direction, num_direction, kernel_size=(width, height), groups=num_direction)
        self.conv = nn.Sequential(
            nn.Conv2d(num_direction, num_direction // 2, (1, 1)),
            nn.Conv2d(num_direction // 2, num_direction, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_depthwise = self.rtgb_depthwise(x)
        out = self.conv(x_depthwise)
        return out


class RTGB3(nn.Module):
    def __init__(self, num_direction, reduction):
        super(RTGB3, self).__init__()
        self.conv = nn.Conv2d(num_direction, 1, (1, 1), bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.fc = nn.Sequential(
            nn.Linear(num_direction, num_direction // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(num_direction // reduction, num_direction, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        input_x = x
        input_x = input_x.view(b, c, h * w).unsqueeze(1)

        mask = self.conv(x).view(b, 1, h * w)
        mask = self.softmax(mask).unsqueeze(-1)
        y = torch.matmul(input_x, mask).view(b, c)

        return self.fc(y)


class TransformerRTGB(nn.Module):
    def __init__(self, c_in, w_in, h_in, heads, rtgb_method):
        super(TransformerRTGB, self).__init__()
        self.msa_c = nn.Sequential(
            MSA(num_vector=c_in, num_heads_column=c_in // heads, heads_number=heads)

        )
        self.msa_h = nn.Sequential(
            MSA(num_vector=h_in, num_heads_column=h_in // heads, heads_number=heads)
        )
        self.msa_w = nn.Sequential(
            MSA(num_vector=w_in, num_heads_column=w_in // heads, heads_number=heads)
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(c_in, c_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(w_in, w_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(h_in, h_in, (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
        )

        self.layernorm_c0 = nn.LayerNorm(c_in)
        self.layernorm_w0 = nn.LayerNorm(w_in)
        self.layernorm_h0 = nn.LayerNorm(h_in)

        self.layernorm_c = nn.LayerNorm(c_in)
        self.layernorm_w = nn.LayerNorm(w_in)
        self.layernorm_h = nn.LayerNorm(h_in)


        if rtgb_method == "RTGB1":
            self.rtgb_c1 = RTGB1(c_in)
            self.rtgb_h1 = RTGB1(h_in)
            self.rtgb_w1 = RTGB1(w_in)
        elif rtgb_method == "RTGB2":
            self.rtgb_c1 = RTGB2(c_in, h_in, w_in)
            self.rtgb_h1 = RTGB2(h_in, w_in, c_in)
            self.rtgb_w1 = RTGB2(w_in, c_in, h_in)
        elif rtgb_method == "RTGB3":
            self.rtgb_c1 = RTGB3(c_in, 2)
            self.rtgb_h1 = RTGB3(h_in, 2)
            self.rtgb_w1 = RTGB3(w_in, 2)
        else:
            print("Invalid RTGB selection")

    def forward(self, x_in):
        b_in, c_in, w_in, h_in = x_in.shape

        x_c = x_in
        x_w = x_in.permute(0, 2, 1, 3)
        x_h = x_in.permute(0, 3, 2, 1)
        x_c = x_c.permute(0, 3, 2, 1)
        x_c = self.layernorm_c0(x_c)
        out_c, out_pc = self.msa_c(x_c)
        x_h = x_h.permute(0, 2, 3, 1)
        x_h = self.layernorm_h0(x_h)
        out_h, out_ph = self.msa_h(x_h)
        x_w = x_w.permute(0, 2, 3, 1)
        x_w = self.layernorm_w0(x_w)
        out_w, out_pw = self.msa_w(x_w)
        out_c = out_c + out_pc
        out_h = out_h + out_ph
        out_w = out_w + out_pw
        out_c = self.layernorm_c(out_c)
        out_w = self.layernorm_w(out_w)
        out_h = self.layernorm_h(out_h)
        out_c = self.conv_c(out_c.permute(0, 3, 1, 2) + x_c.permute(0, 3, 1, 2))
        out_h = self.conv_h(out_h.permute(0, 3, 1, 2) + x_h.permute(0, 3, 1, 2))
        out_w = self.conv_w(out_w.permute(0, 3, 1, 2) + x_w.permute(0, 3, 1, 2))
        vector_c = self.rtgb_c1(out_c)
        vector_h = self.rtgb_h1(out_h)
        vector_w = self.rtgb_w1(out_w)
        assert torch.isnan(vector_c).sum() == 0, print("////")
        assert torch.isnan(vector_h).sum() == 0, print("////")
        assert torch.isnan(vector_w).sum() == 0, print("////")

        matrix3d = torch.empty(b_in, w_in, h_in, requires_grad=True).cuda()

        for i in range(0, b_in):
            matrix3d[i] = torch.outer(vector_w.view(b_in, -1)[i],
                                      vector_h.view(b_in, -1)[i])
        mat_3d = matrix3d.detach().cpu().numpy()
        for j in range(0, b_in):
            assert torch.isnan(matrix3d[j]).sum() == 0, print("////")
        assert torch.isnan(matrix3d).sum() == 0, print("////")

        matrix4d = torch.empty(b_in, c_in, w_in, h_in, requires_grad=True).cuda()
        for i in range(0, b_in):

            matrix4d[i] = torch.outer(vector_c.view(b_in, -1)[i], matrix3d[i].view(-1)) \
                .view(c_in, w_in, h_in)
        assert torch.isnan(matrix4d).sum() == 0, print("////")

        out = matrix4d
        return out


class TransformerDRTLM(nn.Module):
    def __init__(self, c_in, w_in, h_in, heads_num, rtgb_method, rtgb_num):
        super(TransformerDRTLM, self).__init__()
        self.rtgb1 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.rtgb2 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.rtgb3 = TransformerRTGB(c_in, w_in, h_in, heads_num, rtgb_method)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in * rtgb_num, c_in, (3, 3), padding=1),
        )

    def forward(self, x):

        o1 = self.rtgb1(x)
        o2 = self.rtgb2(x - o1)
        o3 = self.rtgb3(x - o1 - o2)
        o_all = torch.cat((o1, o2,o3), dim=1)


        output = self.conv(o_all)
        return output


class Transformer_CP(nn.Module):
    def __init__(self, w_in, h_in, cp_in_c, heads_num, rtgb_method, rtgb_num):
        super(Transformer_CP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )
        self.transformer_DRTLM = TransformerDRTLM(cp_in_c, w_in, h_in, heads_num, rtgb_method, rtgb_num)
        self.resnet = ResNetBlock2(cp_in_c)

    def forward(self, x):


        x1 = self.conv(x)
        assert torch.isnan(x1).sum() == 0, print("////")

        x2 = self.transformer_DRTLM(x1)
        assert torch.isnan(x2).sum() == 0, print("////")

        out = x2 * x1
        assert torch.isnan(out).sum() == 0, print("////")
        out = self.resnet(out)
        return out

def create_F():
     F =np.array([[2.0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2,  6, 11, 17, 21, 22, 21, 20, 20, 19, 19, 18, 18, 17, 17],
         [1,  1,  1,  1,  1,  1,  2,  4,  6,  8, 11, 16, 19, 21, 20, 18, 16, 14, 11,  7,  5,  3,  2, 2,  1,  1,  2,  2,  2,  2,  2],
         [7, 10, 15, 19, 25, 29, 30, 29, 27, 22, 16,  9,  2,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]])
     for band in range(3):
        div = np.sum(F[band][:])
        for i in range(31):
            F[band][i] = F[band][i]/div;
     return F

R = torch.tensor(create_F()).cuda()

class UDoubleTransformerCP3(nn.Module):
    def __init__(self, c_in, w_in, h_in, cp_in_c, c_out, heads_num, stage, rtgb_method, rtgb_num):
        super(UDoubleTransformerCP3, self).__init__()
        self.stage = stage
        self.conv0 = nn.Conv2d(c_in, cp_in_c, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv00 = nn.Conv2d(c_in, c_out, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_de = nn.Conv2d(cp_in_c, cp_in_c, 4, 2, 1, bias=False)
        self.cp1 = Transformer_CP(w_in // 2, h_in // 2, cp_in_c, heads_num, rtgb_method, rtgb_num)
        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(cp_in_c, cp_in_c, (4, 4), (2, 2), 1, bias=False),
            nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(cp_in_c, cp_in_c, (3, 3), (1, 1), padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

        self.fusion1 = nn.Conv2d(cp_in_c*2, cp_in_c, (1, 1), (1, 1), bias=False)
        self.cp2 = Transformer_CP(w_in // 2, h_in // 2, cp_in_c, heads_num, rtgb_method, rtgb_num)

        self.conv_up1 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.conv_up2 = nn.ConvTranspose2d(cp_in_c, cp_in_c, stride=2, kernel_size=2, padding=0, output_padding=0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(cp_in_c, 32, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(cp_in_c, 32, (3, 3), (1, 1), 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(32, 32, (3, 3), (1, 1), 1, bias=False),
            nn.Conv2d(32, c_out, (3, 3), (1, 1), 1, bias=False),
        )
        self.conv1_fu = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu2 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)
        self.conv1_fu3 = nn.Conv2d(cp_in_c, cp_in_c, (1, 1), (1, 1), bias=False)

    def forward(self, x):

        x0 = self.conv0(x)
        x_de = self.conv_de(x0)
        fea1 = self.cp1(x_de)
        fea = self.bottleneck1(fea1)
        fea = self.fusion1(torch.cat([fea, fea1], dim=1))
        fea2 = self.cp2(fea)
        fea = self.conv1_fu3(fea2)
        fea = self.conv_up1(fea)
        temp_x1 = self.conv1_fu2(fea1)
        temp_x1 = self.conv_up2(temp_x1)
        out = self.conv1(fea + temp_x1)
        temp_x0 = self.conv1_fu(x0)
        out = self.conv2(temp_x0 + out)
        out1 = self.conv00(x)
        out = out1 + out
        return out


if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64).cuda()
    model = UDoubleTransformerCP3(3, 64, 64, 32, 31, 4, 1, "RTGB3",3).cuda()

    out = model(x)
    print(out.shape)
