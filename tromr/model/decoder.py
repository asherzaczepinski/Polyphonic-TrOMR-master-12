from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers.x_transformers import AttentionLayers, TokenEmbedding, AbsolutePositionalEmbedding, Decoder


class ScoreTransformerWrapper(nn.Module):
    def __init__(
        self,
        num_note_tokens,
        num_rhythm_tokens,
        num_pitch_tokens,
        num_lift_tokens,
        max_seq_len,
        attn_layers,
        emb_dim,
        l2norm_embed=False,
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed

        # Embeddings
        self.lift_emb = TokenEmbedding(emb_dim, num_lift_tokens, l2norm_embed=l2norm_embed)
        self.pitch_emb = TokenEmbedding(emb_dim, num_pitch_tokens, l2norm_embed=l2norm_embed)
        self.rhythm_emb = TokenEmbedding(emb_dim, num_rhythm_tokens, l2norm_embed=l2norm_embed)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        # Projection layer
        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        # Output heads
        self.to_logits_lift = nn.Linear(dim, num_lift_tokens)
        self.to_logits_pitch = nn.Linear(dim, num_pitch_tokens)
        self.to_logits_rhythm = nn.Linear(dim, num_rhythm_tokens)
        self.to_logits_note = nn.Linear(dim, num_note_tokens)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        if self.l2norm_embed:
            nn.init.normal_(self.lift_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pitch_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.rhythm_emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
        else:
            nn.init.kaiming_normal_(self.lift_emb.emb.weight)
            nn.init.kaiming_normal_(self.pitch_emb.emb.weight)
            nn.init.kaiming_normal_(self.rhythm_emb.emb.weight)

    def forward(self, rhythms, pitchs, lifts, mask=None, return_hiddens=True, **kwargs):
        x = (
            self.rhythm_emb(rhythms)
            + self.pitch_emb(pitchs)
            + self.lift_emb(lifts)
            + self.pos_emb(rhythms)
        )
        x = self.project_emb(x)

        # Pass through attention layers
        x, _ = self.attn_layers(x, mask=mask, return_hiddens=return_hiddens, **kwargs)
        x = self.norm(x)

        # Output logits
        out_lifts = self.to_logits_lift(x)
        out_pitchs = self.to_logits_pitch(x)
        out_rhythms = self.to_logits_rhythm(x)
        out_notes = self.to_logits_note(x)
        return out_rhythms, out_pitchs, out_lifts, out_notes, x


def top_k(logits, thres=0.9):
    """Apply top-k filtering to logits."""
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


class ScoreDecoder(nn.Module):
    def __init__(self, transformer, noteindexes, num_rhythmtoken, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = transformer
        self.max_seq_len = transformer.max_seq_len

        # Note mask for rhythm consistency
        note_mask = torch.zeros(num_rhythmtoken)
        note_mask[noteindexes] = 1
        self.note_mask = nn.Parameter(note_mask)

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        nonote_tokens,
        seq_len,
        eos_token=None,
        temperature=1.2,
        filter_thres=0.8,
        repetition_penalty=1.2,
        **kwargs,
    ):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens.unsqueeze(0)

        b, t = start_tokens.shape

        self.net.eval()
        out_rhythm = start_tokens
        out_pitch = nonote_tokens
        out_lift = nonote_tokens

        mask = kwargs.pop("mask", None)
        if mask is None:
            mask = torch.full_like(out_rhythm, True, dtype=torch.bool, device=device)

        prev_tokens = None  # Track previous tokens for repetition penalty

        for _ in range(seq_len):
            mask = mask[:, -self.max_seq_len:]
            x_lift = out_lift[:, -self.max_seq_len:]
            x_pitch = out_pitch[:, -self.max_seq_len:]
            x_rhythm = out_rhythm[:, -self.max_seq_len:]

            rhythmsp, pitchsp, liftsp, notesp, _ = self.net(
                x_rhythm, x_pitch, x_lift, mask=mask, **kwargs
            )

            # Apply top-k filtering
            filtered_lift_logits = top_k(liftsp[:, -1, :], thres=filter_thres)
            filtered_pitch_logits = top_k(pitchsp[:, -1, :], thres=filter_thres)
            filtered_rhythm_logits = top_k(rhythmsp[:, -1, :], thres=filter_thres)

            # Apply repetition penalty
            if prev_tokens is not None:
                # Convert indices to int64 type for scatter operations
                prev_tokens = prev_tokens.to(torch.int64)
                # Create the penalty tensor with the same dtype as filtered_rhythm_logits
                penalty = torch.full_like(prev_tokens, -repetition_penalty, dtype=filtered_rhythm_logits.dtype)
                filtered_rhythm_logits.scatter_add_(
                    1, prev_tokens, penalty
                )

            # Sample from the distribution
            lift_probs = F.softmax(filtered_lift_logits / temperature, dim=-1)
            pitch_probs = F.softmax(filtered_pitch_logits / temperature, dim=-1)
            rhythm_probs = F.softmax(filtered_rhythm_logits / temperature, dim=-1)

            lift_sample = torch.multinomial(lift_probs, 1)
            pitch_sample = torch.multinomial(pitch_probs, 1)
            rhythm_sample = torch.multinomial(rhythm_probs, 1)

            # Track previous tokens for repetition penalty
            prev_tokens = rhythm_sample

            # Update outputs
            out_lift = torch.cat((out_lift, lift_sample), dim=-1)
            out_pitch = torch.cat((out_pitch, pitch_sample), dim=-1)
            out_rhythm = torch.cat((out_rhythm, rhythm_sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            # Stop if EOS token is generated
            if eos_token is not None and (out_rhythm == eos_token).all(dim=-1).any():
                break

        out_lift = out_lift[:, t:]
        out_pitch = out_pitch[:, t:]
        out_rhythm = out_rhythm[:, t:]

        self.net.train(was_training)
        return out_rhythm, out_pitch, out_lift


def get_decoder(args):
    return ScoreDecoder(
        ScoreTransformerWrapper(
            num_note_tokens=args.num_note_tokens,
            num_rhythm_tokens=args.num_rhythm_tokens,
            num_pitch_tokens=args.num_pitch_tokens,
            num_lift_tokens=args.num_lift_tokens,
            max_seq_len=args.max_seq_len,
            emb_dim=args.decoder_dim,
            attn_layers=Decoder(
                dim=args.decoder_dim,
                depth=args.decoder_depth,
                heads=args.decoder_heads,
                **args.decoder_args,
            ),
        ),
        pad_value=args.pad_token,
        num_rhythmtoken=args.num_rhythmtoken,
        noteindexes=args.noteindexes,
    )