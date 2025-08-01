import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.seq = seq

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Tokenize
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq - len(enc_input_tokens) - 2  # <s>, </s>
        dec_num_padding_tokens = self.seq - len(dec_input_tokens) - 1  # <s>

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            return self.__getitem__(idx + 1)

        # Build encoder input: <s> ... </s> + pad
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Build decoder input: <s> ... + pad
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        # Build label: ... </s> + pad
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
        ], dim=0)

        assert encoder_input.size(0) == self.seq
        assert decoder_input.size(0) == self.seq
        assert label.size(0) == self.seq

        # Key padding masks (batch_first -> (1, seq))
        encoder_key_padding_mask = (encoder_input == self.pad_token) # (seq)
        decoder_key_padding_mask = (decoder_input == self.pad_token)  # (seq)

        # Causal tgt_mask (shared across batch): (seq, seq)
        seq_len = decoder_input.size(0)
        tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

        return {
            "encoder_input": encoder_input,  # (seq)
            "decoder_input": decoder_input,  # (seq)
            "label": label.unsqueeze(0),                  # (1, seq)
            "encoder_key_padding_mask": encoder_key_padding_mask,  # (seq)
            "decoder_key_padding_mask": decoder_key_padding_mask,  # (seq)
            "tgt_mask": tgt_mask,                                   # (seq, seq)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0