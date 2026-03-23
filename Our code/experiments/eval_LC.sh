#SIMPLE-QA
#gemma3-base
python -m evaluating_LC qa_model=gemma-3-1b-it confidence_extractor=linguistic_confidence 
python -m evaluating_LC qa_model=gemma-3-1b-it confidence_extractor=linguistic_confidence_uncertainty 

python -m evaluating_LC qa_model=gemma-3-12b-it confidence_extractor=linguistic_confidence 
python -m evaluating_LC qa_model=gemma-3-12b-it confidence_extractor=linguistic_confidence_uncertainty

python -m evaluating_LC qa_model=gemma-3-27b-it confidence_extractor=linguistic_confidence 
python -m evaluating_LC qa_model=gemma-3-27b-it confidence_extractor=linguistic_confidence_uncertainty

#gemma3-LoRA_SFT
python -m evaluating_LC qa_model=gemma-3-1b-it-LoRA_SFT confidence_extractor=linguistic_confidence

python -m evaluating_LC qa_model=gemma-3-12b-it-LoRA_SFT confidence_extractor=linguistic_confidence

python -m evaluating_LC qa_model=gemma-3-27b-it-LoRA_SFT confidence_extractor=linguistic_confidence 


#NQ-OPEN
#gemma3-base
python -m evaluating_LC qa_model=gemma-3-1b-it confidence_extractor=linguistic_confidence dataset=nq_open
python -m evaluating_LC qa_model=gemma-3-1b-it confidence_extractor=linguistic_confidence_uncertainty dataset=nq_open

python -m evaluating_LC qa_model=gemma-3-12b-it confidence_extractor=linguistic_confidence dataset=nq_open
python -m evaluating_LC qa_model=gemma-3-12b-it confidence_extractor=linguistic_confidence_uncertainty dataset=nq_open

python -m evaluating_LC qa_model=gemma-3-27b-it confidence_extractor=linguistic_confidence dataset=nq_open
python -m evaluating_LC qa_model=gemma-3-27b-it confidence_extractor=linguistic_confidence_uncertainty dataset=nq_open

#gemma3-LoRA_SFT
python -m evaluating_LC qa_model=gemma-3-1b-it-LoRA_SFT confidence_extractor=linguistic_confidence dataset=nq_open

python -m evaluating_LC qa_model=gemma-3-12b-it-LoRA_SFT confidence_extractor=linguistic_confidence dataset=nq_open

python -m evaluating_LC qa_model=gemma-3-27b-it-LoRA_SFT confidence_extractor=linguistic_confidence dataset=nq_open
