from termcolor import colored

from referit3d.models.backbone.dgcnn import DGCNN
from referit3d.models.backbone.lstm_encoder import LSTMEncoder
from referit3d.models.backbone.mlp import MLP
from referit3d.models.backbone.word_embeddings import load_glove_pretrained_embedding, make_pretrained_embedding

try:
    from referit3d.models.backbone.point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None
    msg = colored('Pnet++ is not found. Hence you cannot run all models. Install it via '
                  'external_tools (see README.txt there).', 'red')
    print(msg)

from .baseline_text_classifier import TextClassifier
