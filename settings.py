import collections
import mmh3

ModelSettingsBase = collections.namedtuple(
    "ModelSettings", [
        "max_page_number",
        "font_hash_size",
        "tokens_per_batch",
        "minimum_token_frequency",
        "glove_vectors"
    ]
)

class ModelSettings(ModelSettingsBase):
    def __hash__(self):
        return mmh3.hash(repr(self))

default_model_settings = ModelSettings(
    max_page_number=3,
    font_hash_size=1024 * 4,
    tokens_per_batch=32 * 1024,
    minimum_token_frequency=10,
    glove_vectors="/net/nfs.corp/s2-research/glove/glove.6B.100d.txt.gz"
)
