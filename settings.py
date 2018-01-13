import collections
import mmh3

ModelSettingsBase = collections.namedtuple(
    "ModelSettings", [
        "max_page_number",
        "font_hash_size",
        "tokens_per_batch",
        "minimum_token_frequency",
        "glove_vectors",
        "layer_1_dropout",
        "layer_2_dropout",
    ]
)

class ModelSettings(ModelSettingsBase):
    def __hash__(self):
        return mmh3.hash(repr(self))

default_model_settings = ModelSettings(
    max_page_number=50,
    font_hash_size=1024 * 4,
    tokens_per_batch=32 * 1024,
    minimum_token_frequency=10,
    layer_1_dropout=0.0,
    layer_2_dropout=0.0,
    glove_vectors="model/glove.6B.100d.txt.gz"
)
