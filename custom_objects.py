# custom_objects.py
# Ce fichier contient la définition de la classe personnalisée pour qu'elle soit facilement importable.

import logging
import numpy as np
from sklearn.feature_selection import SelectFromModel

class RobustSelectFromModel(SelectFromModel):
    """
    Wrapper autour de SelectFromModel qui garantit la sélection d'au moins une caractéristique.
    """
    def _get_support_mask(self):
        try:
            mask = super()._get_support_mask()
            if not mask.any():
                logging.warning("Le seuil de SelectFromModel a supprimé toutes les caractéristiques. Conservation de la meilleure caractéristique.")
                importances = None
                estimator = self.estimator_
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                elif hasattr(estimator, "coef_"):
                    importances = np.abs(estimator.coef_[0]) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)

                if importances is not None:
                    mask[np.argmax(importances)] = True
                else:
                    mask[0] = True # Fallback
            return mask
        except Exception as e:
            logging.error(f"Erreur dans RobustSelectFromModel: {e}. Maintien de toutes les caractéristiques.")
            return super()._get_support_mask()
