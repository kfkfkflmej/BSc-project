#gemma3-base
python -m evaluating_on_benchmarks qa_model=gemma-3-1b-it
python -m evaluating_on_benchmarks qa_model=gemma-3-12b-it
python -m evaluating_on_benchmarks qa_model=gemma-3-27b-it

#gemma3-LoRA_SFT
python -m evaluating_on_benchmarks qa_model=gemma-3-1b-it-LoRA_SFT
python -m evaluating_on_benchmarks qa_model=gemma-3-12b-it-LoRA_SFT
python -m evaluating_on_benchmarks qa_model=gemma-3-27b-it-LoRA_SFT