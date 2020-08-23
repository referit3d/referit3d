import torch
import argparse
from torch import nn
from collections import defaultdict

from . import DGCNN
from .default_blocks import *
from .utils import get_siamese_features
from ..in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class ReferIt3DNet(nn.Module):
    """
    A neural listener for segmented 3D scans based on graph-convolutions.
    """

    def __init__(self,
                 args,
                 object_encoder,
                 language_encoder,
                 graph_encoder,
                 object_language_clf,
                 object_clf=None,
                 language_clf=None):
        """
        Parameters have same meaning as in Base3DListener.

        @param args: the parsed arguments
        @param object_encoder: encoder for each segmented object ([point-cloud, color]) of a scan
        @param language_encoder: encoder for the referential utterance
        @param graph_encoder: the graph net encoder (DGCNN is the used graph encoder)
        given geometry is the referred one (typically this is an MLP).
        @param object_clf: classifies the object class of the segmented (raw) object (e.g., is it a chair? or a bed?)
        @param language_clf: classifies the target-class type referred in an utterance.
        @param object_language_clf: given a fused feature of language and geometry, captures how likely it is that the
        """

        super().__init__()

        # The language fusion method (either before the graph encoder, after, or in both ways)
        self.language_fusion = args.language_fusion

        # Encoders
        self.object_encoder = object_encoder
        self.language_encoder = language_encoder
        self.graph_encoder = graph_encoder

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf
        self.object_language_clf = object_language_clf

    def __call__(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = objects_features
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        # Get feature for utterance
        n_objects = batch['objects'].size(1)
        lang_features = self.language_encoder(batch['tokens'])
        lang_features_expanded = torch.unsqueeze(lang_features, -1).expand(-1, -1, n_objects).transpose(
            2, 1)  # B X N_Objects x lang-latent-dim

        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(lang_features)

        # Start graph encoding
        graph_visual_in_features = objects_features
        if self.language_fusion == 'before' or self.language_fusion == 'both':
            graph_in_features = torch.cat([graph_visual_in_features, lang_features_expanded], dim=-1)
        else:
            graph_in_features = graph_visual_in_features

        graph_out_features = self.graph_encoder(graph_in_features)

        if self.language_fusion in ['after', 'both']:
            final_features = torch.cat([graph_out_features, lang_features_expanded], dim=-1)
        else:
            final_features = graph_out_features

        result['logits'] = get_siamese_features(self.object_language_clf, final_features, torch.cat)

        return result


def instantiate_referit3d_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim

    # make an object (segment) encoder for point-clouds with color
    if args.object_encoder == 'pnet_pp':
        object_encoder = single_object_encoder(geo_out_dim)
    else:
        raise ValueError('Unknown object point cloud encoder!')

    # Optional, make a bbox encoder
    object_clf = None
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)

    language_clf = None
    if args.lang_cls_alpha > 0:
        print('Adding a text-classification loss.')
        language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)
        # typically there are less active classes for text, but it does not affect the attained text-clf accuracy.

    # make a language encoder
    lang_encoder = token_encoder(vocab=vocab,
                                 word_embedding_dim=args.word_embedding_dim,
                                 glove_emb_file=args.glove_file,
                                 lstm_n_hidden=lang_out_dim,
                                 word_dropout=args.word_dropout,
                                 random_seed=args.random_seed)

    if args.model.startswith('referIt3DNet'):
        # we will use a DGCNN.
        print('Instantiating a classic DGCNN')

        graph_in_dim = geo_out_dim
        obj_lang_clf_in_dim = args.graph_out_dim

        if args.language_fusion in ['before', 'both']:
            graph_in_dim += lang_out_dim

        if args.language_fusion in ['after', 'both', 'all']:
            obj_lang_clf_in_dim += lang_out_dim

        graph_encoder = DGCNN(initial_dim=graph_in_dim,
                              out_dim=args.graph_out_dim,
                              k_neighbors=args.knn,
                              intermediate_feat_dim=args.dgcnn_intermediate_feat_dim,
                              subtract_from_self=True)

        object_language_clf = object_lang_clf(obj_lang_clf_in_dim)

        model = ReferIt3DNet(
            args=args,
            object_encoder=object_encoder,
            language_encoder=lang_encoder,
            graph_encoder=graph_encoder,
            object_clf=object_clf,
            language_clf=language_clf,
            object_language_clf=object_language_clf)
    else:
        raise NotImplementedError('Unknown listener model is requested.')

    return model
