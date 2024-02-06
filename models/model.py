from torch import nn
from .modeling_bart import BartForMultimodalGeneration


class CommentGenerator_fusion(nn.Module):
    def __init__(self, max_comment_len=200, num_beams=4):
        super(CommentGenerator_fusion, self).__init__()
        self.bart = BartForMultimodalGeneration.from_pretrained("facebook/bart-base",
                                                                fusion_layers=[4, 5],  # [4,5]
                                                                use_forget_gate=False,  # [True]
                                                                dim_common=768,  # 256
                                                                n_attn_heads=1)
        self.max_comment_len = max_comment_len
        self.num_beams = num_beams

    def forward(self, wav_embs, input_ids_lyrics, attention_mask_lyrics=None, input_ids_comments=None):
        if self.training:
            output = self.bart(input_ids=input_ids_lyrics,
                               attention_mask=attention_mask_lyrics,
                               labels=input_ids_comments,
                               music_features=wav_embs,
                               output_hidden_states=True,
                               return_dict=True)
            return output
        else:
            output_ids = self.bart.generate(inputs=input_ids_lyrics,
                                            music_features=wav_embs,
                                            max_length=self.max_comment_len,
                                            early_stopping=True,
                                            do_sample=True,
                                            num_beams=self.num_beams,
                                            top_p=0.92
                                            )
            return output_ids

    def generate(self, wav_embs, input_ids_lyrics):
        output_ids = self.bart.generate(input_ids_lyrics,
                                        max_length=self.max_comment_len,
                                        early_stopping=True,
                                        do_sample=True,
                                        music_features=wav_embs)
        return output_ids
