from typing import List, Optional, Tuple, Union
import random

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import datasets
from transformers.utils import logging
from transformers.utils import (
    add_end_docstrings,
    replace_return_docstrings,
)
from transformers import AutoTokenizer, BartTokenizerFast
from transformers.models.bart.modeling_bart import (
    BartPretrainedModel,
    BaseModelOutput,
    Seq2SeqModelOutput,
    Seq2SeqLMOutput,
    BartConfig,
    BartEncoder,
    BartDecoder,
    BART_INPUTS_DOCSTRING,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    BART_GENERATION_EXAMPLE,
    shift_tokens_right,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from dataclasses import dataclass, field
from transformers import Seq2SeqTrainingArguments, HfArgumentParser


@dataclass
class RunArguments:
    model_name: str = field(default="facebook/bart-large")
    data_name: str = field(default="samsum")
    ctr_mode: str = field(default="baseline")
    lamda: Optional[float] = field(default=0.08)
    batch_size: int = field(default=8)
    set_seed: int = field(default=100)


# arguements parser from shell
parser = HfArgumentParser((Seq2SeqTrainingArguments, RunArguments))
training_args, run_args = parser.parse_args_into_dataclasses()

# arguments from shell
if run_args.ctr_mode == "baseline":
    ctr_mode = 0
elif run_args.ctr_mode == "speaker":
    ctr_mode = 1
elif run_args.ctr_mode == "topic":
    ctr_mode = 2
elif run_args.ctr_mode == "multi":
    ctr_mode = 3
lamda = run_args.lamda
model_name = run_args.model_name
batch_size = run_args.batch_size
set_seed = run_args.set_seed
cluster_mode = 0

device = torch.device("cuda")
print(f"device : {device}")
print("Current cuda device:", torch.cuda.current_device())
print("Count of using GPUs:", torch.cuda.device_count())
logger = logging.get_logger(__name__)

# seed fix
seed = set_seed
random.seed(seed)

# PyTorch seed fix
torch.manual_seed(seed)

# NumPy seed fix
np.random.seed(seed)


# add ctr_loss(Type : torch.FloatTensor, Default : None) = Contrastive Learning Loss(Speaker + Topic)
@dataclass
class CustomSeq2SeqLMOutput(Seq2SeqLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    ctr_loss: torch.FloatTensor = None


# add ctr_speaker_loss(Type : torch.FloatTensor, Default : None) = Speaker-Aware Contrastive Learning Loss
# add ctr_topic_loss(Type : torch.FloatTensor, Default : None) = Topic-Aware Contrastive Learning Loss
@dataclass
class CustomSeq2SeqModelOutput(Seq2SeqModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    ctr_speaker_loss: torch.FloatTensor = None
    ctr_topic_loss: torch.FloatTensor = None


class BartModel(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.num_try = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # enc_speaker : Speaker tokens' Encoder Representations from Huggingface BartModel Encoder
    # ctr_margin : Sigma of Contrastive Learning fomula
    # speaker_input_dis : for discirminating what token is a speaker token
    def speaker_aware(self, enc_speaker, ctr_margin, speaker_input_ids, bench_speaker):
        enc_negative, enc_positive = [], []

        # 多少轮对话
        num_turn = enc_speaker.shape[0]
        num_speaker_unique = len(list(set([int(i[0]) for i in speaker_input_ids])))

        ctr_speaker_loss_means = []
        for speaker in range(1, num_speaker_unique):
            bench_speaker = speaker
            turns = range(1, num_turn)
            for i in turns:
                if torch.eq(speaker_input_ids[i][0], speaker_input_ids[bench_speaker][0]):
                    # same
                    enc_positive.append(enc_speaker[i])
                else:
                    # diff
                    enc_negative.append(enc_speaker[i])

            if len(enc_positive) > 0 and len(enc_negative) > 0:
                relu = nn.ReLU()

                positive_sample_l2 = torch.stack(
                    [
                        torch.dist(positive, enc_speaker[bench_speaker], p=2.0)
                        for positive in enc_positive
                    ]
                )
                negative_sample_l2 = torch.stack(
                    [
                        torch.dist(negative, enc_speaker[bench_speaker], p=2.0)
                        for negative in enc_negative
                    ]
                )

                ctr_speaker_loss_lists = []
                for negative_sample in negative_sample_l2:
                    for positive_sample in positive_sample_l2:
                        softmax_sim_out = nn.functional.softmax(
                            torch.stack([1 - positive_sample, 1 - negative_sample]), dim=0
                        )
                        positive_softmax = softmax_sim_out[0]
                        negative_softmax = softmax_sim_out[1]
                        ctr_speaker_loss_lists.append(
                            relu(ctr_margin - (positive_softmax - negative_softmax))
                        )
                ctr_speaker_loss_list = torch.stack(ctr_speaker_loss_lists)
                ctr_speaker_loss = torch.mean(ctr_speaker_loss_list)
                ctr_speaker_loss_means.append(ctr_speaker_loss)
            else:
                ctr_speaker_loss = torch.zeros(1, device=device)
                return ctr_speaker_loss

        if len(ctr_speaker_loss_means) == 0:
            ctr_speaker_loss = torch.zeros(1, device=device)
            return ctr_speaker_loss
        else:
            ctr_speaker_loss_result = torch.stack(ctr_speaker_loss_means)
            return ctr_speaker_loss_result

    def topic_aware(self, enc_utterance, ctr_margin, cluster_mode):
        df = pd.DataFrame()
        num_turn = len(enc_utterance)
        enc_rep = [rep.cpu().detach().numpy() for rep in enc_utterance]

        if num_turn < 3:
            return torch.zeros(1, device=device)

        if cluster_mode == 0:
            num_cluster = 2
            kmeans = KMeans(n_clusters=num_cluster, init="k-means++").fit(
                enc_utterance.cpu().detach().numpy()
            )
            df = pd.DataFrame({"enc_rep": enc_rep, "cluster": kmeans.labels_})
            centroid = torch.Tensor(kmeans.cluster_centers_).to(device)
            relu = nn.ReLU()
            ctr_topic_loss_means = []
            for bench in range(num_cluster):
                positive_idx = df[df["cluster"] == bench][1:].index.to_numpy()
                negative_idx = df[df["cluster"] != bench].index.to_numpy()

                if len(positive_idx) < 1 or len(negative_idx) < 1:
                    return torch.zeros(1, device=device)

                positive = enc_utterance[positive_idx]
                negative = enc_utterance[negative_idx]

                negative_sample_l2 = [torch.dist(n, centroid[bench], p=2.0) for n in negative]
                positive_sample_l2 = [torch.dist(p, centroid[bench], p=2.0) for p in positive]

                ctr_topic_loss_lists = []
                for negative_sample in negative_sample_l2:
                    for positive_sample in positive_sample_l2:
                        softmax_sim_out = nn.functional.softmax(
                            torch.stack([1 - positive_sample, 1 - negative_sample]), dim=0
                        )
                        positive_softmax = softmax_sim_out[0]
                        negative_softmax = softmax_sim_out[1]
                        ctr_topic_loss_lists.append(
                            relu(ctr_margin - (positive_softmax - negative_softmax))
                        )
                ctr_topic_loss_list = torch.stack(ctr_topic_loss_lists)
                ctr_topic_loss = torch.mean(ctr_topic_loss_list)
                ctr_topic_loss_means.append(ctr_topic_loss)

            ctr_topic_loss_result = torch.stack(ctr_topic_loss_means)
            return ctr_topic_loss_result

        elif cluster_mode == 1:
            turn_topics = [0 if i < num_turn // 2 else 1 for i in range(num_turn)]
            centroid = torch.stack(
                [enc_utterance[num_turn // 4], enc_utterance[num_turn // 2 + num_turn // 4]]
            )
            df = pd.DataFrame({"enc_rep": enc_rep, "cluster": turn_topics})

            relu = nn.ReLU()
            ctr_topic_loss_means = []
            for bench in range(num_turn):
                positive_idx = df[df["cluster"] == bench][1:].index.to_numpy()
                negative_idx = df[df["cluster"] != bench].index.to_numpy()
                if len(positive_idx) >= 1 and len(negative_idx) >= 1:
                    positive = enc_utterance[positive_idx]
                    negative = enc_utterance[negative_idx]

                    negative_sample_l2 = [torch.dist(n, centroid[bench], p=2.0) for n in negative]
                    positive_sample_l2 = [torch.dist(p, centroid[bench], p=2.0) for p in positive]

                    ctr_topic_loss_lists = []
                    for negative_sample in negative_sample_l2:
                        for positive_sample in positive_sample_l2:
                            softmax_sim_out = nn.functional.softmax(
                                torch.stack([1 - positive_sample, 1 - negative_sample]), dim=0
                            )
                            positive_softmax = softmax_sim_out[0]
                            negative_softmax = softmax_sim_out[1]
                            ctr_topic_loss_lists.append(
                                relu(ctr_margin - (positive_softmax - negative_softmax))
                            )
                    ctr_topic_loss_list = torch.stack(ctr_topic_loss_lists)
                    ctr_topic_loss = torch.mean(ctr_topic_loss_list)
                    ctr_topic_loss_means.append(ctr_topic_loss)
            ctr_topic_loss_result = torch.stack(ctr_topic_loss_means)
            return ctr_topic_loss_result

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        all_special_ids: Optional[List] = None,
        raw_data: Optional[datasets.dataset_dict.DatasetDict] = None,
        ctr_mode: int = 0,
        cluster_mode: int = 0,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # from modelscope import snapshot_download
        # model_dir = snapshot_download("zimuwangnlp/bart-large")
        # tokenizer = BartTokenizerFast.from_pretrained(model_dir)
        # tokenizer.add_special_tokens({"additional_special_tokens": ["<sep>", ":"]})

        # print(encoder_outputs)
        # print(input_ids)
        if all_special_ids is not None:
            # print(all_special_ids)
            lang_sep = 5  # English
            sep_idx = [
                idx
                for ids, idx in zip(input_ids[0], range(input_ids.shape[1]))
                if ids == all_special_ids[lang_sep]
            ]
            # sep的位置
            # print(sep_idx)
            speaker_idx = []
            # 找到speaker的全称
            for idx in sep_idx:
                ids = idx
                while (
                    ids < len(input_ids[0]) and input_ids[0][ids] != all_special_ids[lang_sep + 1]
                ):
                    ids += 1
                speaker_idx.append([idx + 1, ids])
            
            # print(speaker_idx)
            # assert len(speaker_idx) == len(sep_idx)

            # utterance_idx = [
            #     [start + 3, end]
            #     for start, end in zip(sep_idx[:-1], sep_idx[1:])
            #     if (start + 3) < end
            # ]
            utterance_idx = [
                [start[-1]+1, end]
                for start, end in zip(speaker_idx[:-1], sep_idx[1:])
                if start[-1]+1 < end
            ] 
            utterance_idx.append([speaker_idx[-1][-1]+1, len(input_ids[0])])
            # assert 
            # print(utterance_idx)
            speaker_input_ids = [input_ids[0][i[0]:i[1]] for i in speaker_idx]

            # assert len(utterance_idx) == len(speaker_input_ids)


        if ctr_mode == 0:  # 기존 BART만 Training
            ctr_speaker_loss = torch.zeros(1, device=device)
            ctr_topic_loss = torch.zeros(1, device=device)
        elif ctr_mode == 1:  # BART + Spaeker-Aware
            if len(speaker_idx) > 1:
                enc_speaker = [encoder_outputs[0][0][i[0]:i[1]] for i in utterance_idx]

                if len(speaker_input_ids) != len(enc_speaker):
                    min_l = min(len(speaker_input_ids),len(enc_speaker))+1
                    enc_speaker = enc_speaker[:min_l]
                    speaker_input_ids = speaker_input_ids[:min_l]

                # 按行堆叠tensor
                mean_speaker = torch.row_stack([torch.mean(i, 0) for i in enc_speaker])

                # Speaker-Aware
                ctr_speaker_loss = self.speaker_aware(
                    enc_speaker=mean_speaker,  # speaker의 representation list
                    ctr_margin=1,  # ctrastive learning 시, margin 값
                    speaker_input_ids=speaker_input_ids,  # Dialogue 안 Speaker Token들의 input_ids list
                    bench_speaker=0,  # P01을 기준점 = 0번째 Speaker
                )
                ctr_topic_loss = torch.zeros(1, device=device)
            else:
                ctr_speaker_loss = torch.zeros(1, device=device)
                ctr_topic_loss = torch.zeros(1, device=device)

        elif ctr_mode == 2:  # koBART + Topic-view
            if len(speaker_idx) > 1:
                # Utterance의 Encoder Representation -> Mean Pooling
                enc_utterance = [encoder_outputs[0][0][i[0]:i[1]] for i in utterance_idx]
                mean_utterance = torch.row_stack([torch.mean(i, 0) for i in enc_utterance])

                # Topic-Aware
                ctr_topic_loss = self.topic_aware(
                    enc_utterance=mean_utterance,  # Mean Pooling한 utterance의 representation list
                    ctr_margin=1,  # ctrastive learning 시, margin 값
                    cluster_mode=cluster_mode,  # 0=Kmeans, 1=Sequential
                )
                ctr_speaker_loss = torch.zeros(1, device=device)
            else:
                ctr_speaker_loss = torch.zeros(1, device=device)
                ctr_topic_loss = torch.zeros(1, device=device)

        elif ctr_mode == 3:  # koBART + Spaeker-Aware + Topic-Aware
            if len(speaker_idx) > 1:
                # Speaker의 Encoder Representation -> Mean Pooling
                enc_speaker = [encoder_outputs[0][0][i[0]:i[1]] for i in speaker_idx]
                mean_speaker = torch.row_stack([torch.mean(i, 0) for i in enc_speaker])

                # Speaker-Aware
                ctr_speaker_loss = self.speaker_aware(
                    enc_speaker=mean_speaker,  # speaker의 representation list
                    ctr_margin=1,  # ctrastive learning 시, margin 값
                    speaker_input_ids=speaker_input_ids,  # Dialogue 안 Speaker Token들의 input_ids list
                    bench_speaker=0,  # P01을 기준점 = 0번째 Speaker
                )

                # Utterance의 Encoder Representation -> Mean Pooling
                enc_utterance = [encoder_outputs[0][0][i[0]:i[1]] for i in utterance_idx]
                mean_utterance = torch.row_stack([torch.mean(i, 0) for i in enc_utterance])

                # Topic-Aware
                ctr_topic_loss = self.topic_aware(
                    enc_utterance=mean_utterance,  # Mean Pooling한 utterance의 representation list
                    ctr_margin=1,  # ctrastive learning 시, margin 값
                    cluster_mode=cluster_mode,  # 0=Kmeans, 1=Sequential
                )
            else:
                ctr_speaker_loss = torch.zeros(1, device=device)
                ctr_topic_loss = torch.zeros(1, device=device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return CustomSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            ctr_speaker_loss=ctr_speaker_loss,
            ctr_topic_loss=ctr_topic_loss,
        )


class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"lm_head.weight",
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        all_special_ids: Optional[List] = None,
        return_dict: Optional[bool] = None,
        raw_data: Optional[datasets.dataset_dict.DatasetDict] = None,
        ctr_mode: int = 0,
        cluster_mode: int = 0,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            all_special_ids=all_special_ids,
            raw_data=raw_data,
            ctr_mode=ctr_mode,
            cluster_mode=cluster_mode,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return CustomSeq2SeqLMOutput(
            loss=masked_lm_loss,
            ctr_loss=torch.mean(outputs.ctr_speaker_loss) + torch.mean(outputs.ctr_topic_loss),
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
