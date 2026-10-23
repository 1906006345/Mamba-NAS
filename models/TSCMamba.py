import torch
import torch.nn.functional as F
from mamba_ssm import Mamba
import copy
from einops.layers.torch import Rearrange

class Model(torch.nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs=configs
        self.configs_copy = copy.deepcopy(self.configs)
        if self.configs.task_name=="classification":
            self.patcher=ConversionLayer(
                d_in=self.configs.enc_in,
                l_in=self.configs.seq_len,
                l_out=self.configs.projected_space,
                d_out=self.configs.enc_in,
                im_size=self.configs.rescale_size,
                patch_size=self.configs.patch_size)
            if self.configs.no_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space)
            elif self.configs.half_rocket==1:
                self.projector=torch.nn.Linear(self.configs.seq_len,self.configs.projected_space//2)

            self.ln=torch.nn.LayerNorm(self.configs.projected_space*3)
            self.dropout=torch.nn.Dropout(self.configs.dropout)
            self.learnable_focus=torch.nn.Parameter(torch.tensor([self.configs.initial_focus]))
            self.gelu=torch.nn.GELU()
            self.mamba1 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.projected_space*3,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.d_conv,
                      expand=self.configs.expand) for _ in range(self.configs.num_mambas)
                ])
            self.mamba2 = torch.nn.ModuleList([
                Mamba(d_model=self.configs.enc_in,
                      d_state=self.configs.d_state,
                      d_conv=self.configs.d_conv,
                      expand=self.configs.expand) for _ in range(self.configs.num_mambas)
                ])

            self.flatten=torch.nn.Flatten(start_dim=1)
            self.classifier=torch.nn.Sequential(
                torch.nn.Linear(self.configs.projected_space*3,(self.configs.projected_space*3)//2),
                torch.nn.Dropout(self.configs.dropout),
                torch.nn.Linear((self.configs.projected_space*3)//2,self.configs.num_class)        
            )
            

    def classification(self,x_cwt,batch_x_features):    
        x_patched=self.patcher(x_cwt)
        if self.configs.no_rocket==1:
            x_projected=self.projector(batch_x_features)
        else:
            if self.configs.half_rocket==0:
                x_projected=batch_x_features
            else:
                if batch_x_features.shape[-1] == self.configs.seq_len:
                    x_projected=torch.cat([batch_x_features[:,:,:self.configs.projected_space//2],
                                           self.projector(batch_x_features[:,:,self.configs.projected_space:])
                                        ],dim=2)
                else:
                    x_projected = F.interpolate(
                        batch_x_features,
                        size=self.configs.projected_space,
                        mode='linear',
                        align_corners=False,
                    )
        if self.configs.additive_fusion==1:
            x_fused=(self.learnable_focus*x_projected)+(2.0-self.learnable_focus)*x_patched
        else:
            x_fused=(self.learnable_focus*x_projected)*(2.0-self.learnable_focus)*x_patched
            
        
        x_fused=self.gelu(x_fused)
        concatenated_x=torch.cat([x_patched,x_fused,x_projected],dim=2)
        concatenated_x=self.ln(concatenated_x)

        if self.configs.num_mambas!=0:
            x1=concatenated_x.clone()

            for i in range(self.configs.num_mambas):
                x1=self.mamba1[i](x1)+x1.clone()

            if self.configs.only_forward_scan==0:
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])
                x1_flipped=concatenated_x.clone()
                for i in range(self.configs.num_mambas):
                    x1_flipped=self.mamba1[i](x1_flipped)+x1_flipped.clone()     
                if self.configs.reverse_flip==0:           
                    x1=x1+x1_flipped
                elif self.configs.reverse_flip==1:
                    x1=x1+torch.flip(x1_flipped,dims=[self.configs.flip_dir])
                concatenated_x=torch.flip(concatenated_x,dims=[self.configs.flip_dir])

            concatenated_x=torch.permute(concatenated_x,(0,2,1))
            x2=concatenated_x.clone()
            for i in range(self.configs.num_mambas):
                x2=self.mamba2[i](x2)+x2.clone()
            x2=torch.permute(x2,(0,2,1))
            
            if self.configs.only_forward_scan==0:
                x2_flipped=torch.flip(concatenated_x.clone(),dims=[self.configs.flip_dir])
                for i in range(self.configs.num_mambas):
                    x2_flipped=self.mamba2[i](x2_flipped)+x2_flipped.clone()     

                x2_flipped=torch.permute(x2_flipped,(0,2,1))
                if self.configs.reverse_flip==0:
                    x2=x2+x2_flipped 
                elif self.configs.reverse_flip==1:
                    x2=x2+torch.flip(x2_flipped,dims=[self.configs.flip_dir])
            x3=x2+x1
        else:
            x3=concatenated_x
        if self.configs.max_pooling==0:
            x3 = x3.mean(1)
        else:
            x3, _ = x3.max(1)
        x3=self.flatten(x3)
        x_logits=self.classifier(x3)
        return x_logits

    def _sequence_inputs(self, x_enc, padding_mask=None):
        """Convert the repository's [B, L, C] batch to the source inputs.

        The original implementation received a CWT image and a projected
        feature tensor from its data preprocessing code. The unified UEA
        runner supplies only a padded time series, so we construct an
        image-shaped interpolation and a channel-wise projected sequence here.
        This keeps the source architecture and its fusion path unchanged.
        """
        if x_enc.ndim != 3:
            raise ValueError('Expected x_enc with shape [B, L, C]')
        if padding_mask is not None:
            mask = padding_mask.to(device=x_enc.device).bool()
            x_enc = x_enc * mask.unsqueeze(-1).to(x_enc.dtype)
        sequence_features = x_enc.permute(0, 2, 1)
        projected_features = F.interpolate(
            sequence_features,
            size=self.configs.projected_space,
            mode='linear',
            align_corners=False,
        )
        if self.configs.no_rocket==1:
            x_features = sequence_features
        elif self.configs.half_rocket==1:
            # The source implementation receives half precomputed ROCKET
            # features. For the standard sequence loader, use direct temporal
            # interpolation for that half and project the raw sequence for the
            # other half inside ``classification``.
            x_features = projected_features
        else:
            x_features = projected_features
        image_values = F.interpolate(
            x_enc.permute(0, 2, 1),
            size=self.configs.rescale_size * self.configs.rescale_size,
            mode='linear',
            align_corners=False,
        )
        x_cwt = image_values.reshape(
            x_enc.shape[0],
            x_enc.shape[2],
            self.configs.rescale_size,
            self.configs.rescale_size,
        )
        return x_cwt, x_features

    def forward(self, x_enc, padding_mask=None, x_dec=None, x_mark_dec=None, mask=None):
         if self.configs.task_name=='classification':
             x_cwt, x_features = self._sequence_inputs(x_enc, padding_mask)
             return self.classification(x_cwt, x_features)
       

class ConversionLayer(torch.nn.Module):
    def __init__(self,d_in,l_in,d_out,l_out,im_size,patch_size):
        super(ConversionLayer, self).__init__()
        self.d_in=d_in
        self.l_in=l_in
        self.d_out=d_out
        self.l_out=l_out
        self.im_size=im_size
        self.patch_size=patch_size

        self.patch_embedding=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size), padding=0),
            Rearrange('b c h w -> b c (h w)')
        )
        self.projector=torch.nn.Linear((self.im_size//self.patch_size)*(self.im_size//self.patch_size),self.l_out)
        
    def forward(self,x):
        x=self.patch_embedding(x)
        x=self.projector(x)
        return x
