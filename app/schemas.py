from pydantic import BaseModel

class InputData(BaseModel):
    surface_reelle_bati: float
    nombre_pieces_principales: int
    latitude: float
    longitude: float
    has_dependance: int
    nom_commune: str