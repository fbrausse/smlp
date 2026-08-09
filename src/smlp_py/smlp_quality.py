from .ra.quality import Quality
from .smlp_subgroups import SubgroupDiscovery

import logging
import pandas as pd

class SmlpQuality(Quality):
    def __init__(self, logger: logging.Logger,  quality_function: str, sd: SubgroupDiscovery) -> None:
        super().__init__(logger, quality_function)
        self.sd = sd

    def rank(self, feat_names: list[str], feat_df: pd.DataFrame, resp_df: pd.DataFrame, resp_name: str):
        self.logger.info(f"Ranking based on quality metrics with a quality function {self.quality_function}")
        
        resp = resp_df[resp_name]
        result = {
            'Feature': [],
            'TPR': [],
            'PPV': [],
            'Lift': [],
            'ROCAcc': [],
            'NPLR': [],
            'WRAcc': [],
            'F1Score': [],
            'Acc': [],
            'Kappa': [],
            'Score': []
        }
        
        for feature in feat_names:
            feat = feat_df[feature]

            scores = self.sd.feat_resp2opt_scores(feature, feat, resp, [], None, None, 1, 0)

            result['Feature'].append(feature)
            result['TPR'].append(scores['TruPos'] / len((feat == 1)))
            result['PPV'].append(scores['TruPos'] / (scores['TruPos'] + scores['FalPos']))
            result['Lift'].append(scores['Lift'])
            result['ROCAcc'].append(scores['ROCAcc'])
            result['NPLR'].append(scores['NormPosLR'])
            result['WRAcc'].append(scores['WRAcc'])
            result['F1Score'].append(scores['F1Score'])
            result['Acc'].append(scores['Accuracy'])
            result['Kappa'].append(scores['CohenKappa'])
            result['Score'].append(result[self.quality_function][-1])

        result_df = pd.DataFrame(result, columns = ['Feature', 'TPR', 'PPV', 'Lift', 'ROCAcc', 'NPLR', 'WRAcc', 'F1Score', 'Acc', 'Kappa', 'Score'])

        return result_df            