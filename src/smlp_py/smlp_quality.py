from .ra.quality import Quality
from .smlp_subgroups import SubgroupDiscovery

import logging
import pandas as pd

class SmlpQuality(Quality):
    def __init__(self, logger: logging.Logger,  quality_function: str, sd: SubgroupDiscovery) -> None:
        super().__init__(logger, quality_function)
        self.sd = sd

    def compute_metrics(self, feat_names: list[str], feat_df: pd.DataFrame, resp_df: pd.DataFrame, resp_name: str):
        self.logger.info(f"Computing quality metrics")
        
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

        quality_metrics_df = pd.DataFrame(result, columns = ['Feature', 'TPR', 'PPV', 'Lift', 'ROCAcc', 'NPLR', 'WRAcc', 'F1Score', 'Acc', 'Kappa'])

        return quality_metrics_df            

    def rank(self, quality_metrics_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Ranking based on the quality metric {self.quality_function}")

        result_df = pd.DataFrame()

        result_df['Feature'] = quality_metrics_df['Feature']
        result_df['Score'] = quality_metrics_df[self.quality_function]

        return result_df
