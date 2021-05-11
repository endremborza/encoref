import pandas as pd

from encoref import CoReferenceLock, EntitySetPair, RelationPair

esp_pers = EntitySetPair(
    pd.DataFrame({"name": ["frank", "carl", "mark"], "age": [14, 16, 19]}),
    pd.DataFrame(
        {
            "name": ["frankie", "cj", "ross", "dobbie"],
            "age": [13.6, 16.2, 20.1, 14],
        }
    ),
    name="person",
)

esp_hobby = EntitySetPair(
    pd.DataFrame({"name": ["footy", "gaming"]}),
    pd.DataFrame({"name": ["soccer", "playstation"]}),
    name="hobby",
)


esp_event = EntitySetPair(
    pd.DataFrame({"name": ["match", "stream", "party"]}),
    pd.DataFrame({"name": ["footballgame", "live-stream"]}),
    name="event",
)

relp_ph = RelationPair(
    pd.DataFrame({"pers": [1, 0, 1], "thing": [1, 1, 0]}),
    pd.DataFrame({"pers": [0, 1, 1, 2], "thing": [1, 0, 1, 1]}),
    name="ph",
    entity_types_of_columns=["person", "hobby"],
)


relp_he = RelationPair(
    pd.DataFrame({"hobby": [0, 1, 1], "event": [0, 1, 2]}),
    pd.DataFrame({"hobby": [0, 1], "event": [0, 1]}),
    name="he",
)


def get_hobby_crl():
    return CoReferenceLock(
        [esp_pers, esp_hobby, esp_event], [relp_ph, relp_he]
    )
