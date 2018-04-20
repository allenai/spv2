import collections
import mmh3

ModelSettingsBase = collections.namedtuple(
    "ModelSettings", [
        "max_page_number",
        "font_hash_size",
        "tokens_per_batch",
        "embedded_tokens_fraction",
        "glove_vectors"
    ]
)

class ModelSettings(ModelSettingsBase):
    def __hash__(self):
        return mmh3.hash(repr(self))

default_model_settings = ModelSettings(
    max_page_number=50,
    font_hash_size=1024 * 4,
    tokens_per_batch=44 * 1024,
    embedded_tokens_fraction=0.995,
    glove_vectors="model/glove.6B.100d.txt.gz"
)
