import torch
from torch import nn


class Pooling(nn.Module):
    """Methods to obtains sentence embeddings from word vectors. Multiple methods
    can be specificed and their results will be concatenated together.

    Arguments:
        sent_rep_tokens (bool, optional): Use the sentence representation token
                as sentence embeddings. Default is True.
        mean_tokens (bool, optional): Take the mean of all the token vectors in
        each sentence. Default is False.
    """

    def __init__(self, sent_rep_tokens=True, mean_tokens=False, max_tokens=False):
        super(Pooling, self).__init__()

        self.sent_rep_tokens = sent_rep_tokens
        self.mean_tokens = mean_tokens
        self.max_tokens = max_tokens

        # pooling_mode_multiplier = sum([sent_rep_tokens, mean_tokens])
        # self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(
        self,
        word_vectors=None,
        sent_rep_token_ids=None,
        sent_rep_mask=None,
        sent_lengths=None,
        sent_lengths_mask=None,
    ):
        r"""Forward pass of the Pooling nn.Module.

        Args:
            word_vectors (torch.Tensor, optional): Vectors representing words created by
                a ``word_embedding_model``. Defaults to None.
            sent_rep_token_ids (torch.Tensor, optional): See
                :meth:`extractive.ExtractiveSummarizer.forward`. Defaults to None.
            sent_rep_mask (torch.Tensor, optional): See
                :meth:`extractive.ExtractiveSummarizer.forward`. Defaults to None.
            sent_lengths (torch.Tensor, optional): See
                :meth:`extractive.ExtractiveSummarizer.forward`. Defaults to None.
            sent_lengths_mask (torch.Tensor, optional): See
                :meth:`extractive.ExtractiveSummarizer.forward`. Defaults to None.

        Returns:
            tuple: (output_vector, output_mask) Contains the sentence scores and mask as
            ``torch.Tensor``\ s. The mask is either the ``sent_rep_mask`` or
            ``sent_lengths_mask`` depending on the pooling mode used during model initialization.
        """
        output_vectors = []
        output_masks = []

        _,_,dim = word_vectors.size()

        if self.sent_rep_tokens:
            sents_vec = word_vectors[
                torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
            ]
            sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_rep_mask)

        if self.mean_tokens or self.max_tokens:
            # batch_sequences = [
            #     torch.split(word_vectors[idx], list(seg.cpu().numpy()))
            #     for idx, seg in enumerate(sent_lengths)
            # ]
            batch_sequences = []
            for idx, seg in enumerate(sent_lengths):
                split_list = [len_ for len_ in list(seg.cpu().numpy()) if (len_ != 0)]
                # 记录长度补零的个数
                zero_num = sum([len_+1 for len_ in list(seg.cpu().numpy()) if (len_ == 0)])
                # [merge_num, H]
                pad_tensor = torch.zeros(dim).to(word_vectors)
                # print(sum(split_list))
                seg = list(torch.split(word_vectors[idx][:sum(split_list),:],split_list)) + zero_num*[pad_tensor.unsqueeze(0)] 
                batch_sequences.append(seg)

            sents_list = [
                torch.stack(
                    [
                        # the mean with padding ignored
                        (
                            # (sequence.sum(dim=0) / (sequence != 0).sum(dim=0))
                            # if self.mean_tokens
                            # else torch.max(sequence, 0)[0]  # 取每个sent中每一个hidden维度的最大值
                            (sequence.sum(dim=0) / (sequence != 0).sum(dim=0))
                            if self.mean_tokens
                            else sequence.sum(dim=0) # 取每个sent中每一个hidden维度的最大值
                        )
                        # if the sequence contains values that are not zero
                        if ((sequence != 0).sum() != 0)
                        # any tensor with 2 dimensions (one being the hidden size) that has already
                        # been created (will be set to zero from padding)
                        else word_vectors[0, 0].float()
                        # for each sentence
                        for sequence in sequences
                    ],
                    dim=0,
                )
                for sequences in batch_sequences  # for all the sentences in each batch
            ]
            # B, sent_num, H
            sents_vec = torch.stack(sents_list, dim=0)
            # print(sents_vec.size(),sent_lengths_mask.size())
            sents_vec = sents_vec * sent_lengths_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_lengths_mask)

        output_vector = torch.cat(output_vectors, 1)
        output_mask = torch.cat(output_masks, 1)

        return output_vector, output_mask
