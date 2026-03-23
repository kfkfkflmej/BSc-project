from .linguistic_confidence import LinguisticConfidenceExtractor

from omegaconf import DictConfig

class ConfidenceExtractor:
    def __init__(self, confidence_extraction_method_cfg: DictConfig, qa_model_cfg: DictConfig):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.qa_model_cfg = qa_model_cfg
        
        self.confidence_extractor = self.get_confidence_extractor(confidence_extraction_method_cfg.name)
    
    def __call__(self, dataset, pre_runned_batch_info: DictConfig):
        return self.confidence_extractor(dataset, pre_runned_batch_info)
    
    def get_confidence_extractor(self, confidence_extraction_method_name):
        if confidence_extraction_method_name == "linguistic_confidence": 
            return LinguisticConfidenceExtractor(self.confidence_extraction_method_cfg, self.qa_model_cfg)
        elif confidence_extraction_method_name == "linguistic_confidence_uncertainty": 
            return LinguisticConfidenceExtractor(self.confidence_extraction_method_cfg, self.qa_model_cfg, uncertainty=True)
        else:
            raise ValueError(f"Invalid confidence extraction method: {confidence_extraction_method_name}")