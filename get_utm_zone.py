def get_utm_zone(lon: float) -> int:
    """
    Calcule la zone UTM à partir d'une longitude (en degrés).
    Fonctionne pour tout point sur le globe (sauf les pôles).

    Args:
        lon (float): Longitude en degrés (positif pour Est, négatif pour Ouest).

    Returns:
        int: Numéro de la zone UTM (1 à 60).
    """
    return int((lon + 180) // 6) + 1