import collections
import mmh3

ModelSettingsBase = collections.namedtuple(
    "ModelSettings", [
        "max_page_number",
        "token_hash_size",
        "font_hash_size",
        "token_vector_size",
        "batch_size"
    ]
)

class ModelSettings(ModelSettingsBase):
    def __hash__(self):
        return mmh3.hash(repr(self))


default_model_settings = ModelSettings(
    max_page_number=3,
    token_hash_size=1024 * 32,
    font_hash_size=1024 * 4,
    token_vector_size=1024,
    batch_size=4
)
