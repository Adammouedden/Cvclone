import torch
import torch.nn as nn

class GRUSeq2Seq(nn.Module):
    def __init__(self, in_dim=3, hid=64, num_layers=1, out_steps=8):
        super().__init__()
        self.encoder = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(input_size=in_dim, hidden_size=hid, num_layers=num_layers, batch_first=True)
        self.head    = nn.Linear(hid, 2)
        self.out_steps = out_steps

    def forward(self, x_in, dts_out, teacher=None):
        """
        x_in:    [B, T_in,  3]  = (x, y, Δt_norm)
        dts_out: [B, T_out, 1]  (Δt_norm for each forecast step)
        teacher: [B, T_out, 2]  optional (x,y) for teacher forcing
        """
        B = x_in.size(0)
        _, h = self.encoder(x_in)  # h: [num_layers, B, hid]

        last_xy = x_in[:, -1, :2]  # [B,2]
        preds = []
        dec_in = torch.cat([last_xy, dts_out[:, 0, :]], dim=-1).unsqueeze(1)  # [B,1,3]

        for t in range(dts_out.size(1)):
            out, h = self.decoder(dec_in, h)     # out: [B,1,hid]
            xy = self.head(out[:, 0, :])         # [B,2]
            preds.append(xy)

            next_xy = teacher[:, t, :] if (self.training and teacher is not None) else xy
            if t + 1 < dts_out.size(1):
                dec_in = torch.cat([next_xy, dts_out[:, t+1, :]], dim=-1).unsqueeze(1)

        return torch.stack(preds, dim=1)  # [B, T_out, 2]
