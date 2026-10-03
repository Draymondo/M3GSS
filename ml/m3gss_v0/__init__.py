"""Package M3GSS_v0 — premier modele IA de reconstruction x2 du projet M3GSS.

Architecture prevue (voir rapport de conception) :
M3GSS_v0_32x8 — CNN residuel leger :
  entree bicubique RGB -> Conv 3x3 (3->32) -> LeakyReLU(0.1)
  -> 8 blocs residuels (Conv3x3 -> LeakyReLU -> Conv3x3)
  -> Conv trunk (32->32) -> Conv finale (32->3, zero-init)
  -> sortie = bicubique + residu predit, sans BatchNorm
  ~159 000 parametres.

STATUT : environnement prepare uniquement. Le module modele et
l'entrainement ne sont pas encore implementes (etape suivante).
"""

__all__: list[str] = []
