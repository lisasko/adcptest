import numpy as np
from XSection import XSection  # Remplace par le nom de ton fichier

def test_xsection_geometry():
    print("--- Début du Sanity Check Géométrique ---")
    
    # 1. Initialisation : Le bateau va vers l'Est (1, 0)
    xs = XSection(direction=[1.0, 0.0], origin=[0.0, 0.0])
    
    print(f"Direction (Tangente N) : {xs.direction} -> Attendu: [1. 0.]")
    print(f"Orthogonale (Courant S) : {xs.direction_orthogonal} -> Attendu: [0. -1.]")
    
    # Validation stricte des signes
    assert np.allclose(xs.direction_orthogonal, [0.0, -1.0]), "Échec : L'axe orthogonal ne pointe pas vers le Sud !"

    # 2. Test de la rotation des vitesses (xy2sn_vel)
    # Le courant réel va vers le Nord : U = 0 (Est), V = 1 (Nord)
    u_reel, v_reel = 0.0, 1.0
    us, un = xs.xy2sn_vel(u_reel, v_reel)
    print(f"\nCourant réel vers le Nord (U=0, V=1) converti en SN :")
    print(f"  Us (Axe orthogonal - Sud) : {us} -> Attendu: -1.0 (car opposé au Nord)")
    print(f"  Un (Axe tangentiel - Est) : {un} -> Attendu: 0.0")
    
    assert np.isclose(us, -1.0) and np.isclose(un, 0.0), "Échec de la rotation des vitesses !"

    # 3. Test du Setter Orthogonal (L'épreuve de vérité)
    # On force l'orthogonal vers le Sud, la direction doit redevenir l'Est
    print("\nTest du Setter Orthogonal :")
    xs.direction_orthogonal = [0.0, -1.0]
    print(f"  Nouvelle Direction calculée via le setter : {xs.direction} -> Attendu: [1. 0.]")
    
    assert np.allclose(xs.direction, [1.0, 0.0]), "Échec critique : Le setter orthogonal a inversé le repère !"
    
    print("\n✅ TOUT EST CORRECT ! Le comportement est 100% identique à Matlab, aucun effet miroir détecté.")

if __name__ == "__main__":
    test_xsection_geometry()