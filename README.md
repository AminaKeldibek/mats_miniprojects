# mats_miniprojects

1. connect with ssh: copy command from runpod page and replace key with ~/.ssh/runpod
2. scp -P 11627 startup.sh root@205.196.17.27:/root
3. source ./startup.sh
4. git clone https://github.com/AminaKeldibek/alignment_faking_mechanistic.git
5. source scripts/bin/runpod_env.sh
3. scp -P 11627 /Users/akeldibek/Projects/llm_tutorials/alignment_faking_mechanistic/.env root@205.196.17.27:/root/alignment_faking_mechanistic
7. poetry run huggingface-cli login --token $HUGGINGFACE_TOKEN


  Run following scripts:
1. poetry run python scripts/run_vector_extraction.py /workspace/data/llama_405b_vector_extraction_no_reasoning/config.yaml
2. poetry run python scripts/upload_to_huggingface.py /workspace/data/llama_405b_vector_extraction_no_reasoning/config.yaml --monitor --interval 100
3. poetry run python -m alignment_faking_mechanistic.extract_unembedding /workspace/data/llama_405b_vector_extraction_no_reasoning/config.yaml
4. poetry run python scripts/upload_unembedding_to_huggingface.py /workspace/data/llama_405b_vector_extraction_no_reasoning/config.yaml
ls -lh /workspace/activations_output/*_unembedding.h5

scp -P 11627 root@205.196.17.27:/workspace/activations_output/llama_405b_unembedding.h5 ~/Projects/llm_tutorials/alignment_faking_mechanistic/data/




ssh root@205.196.17.27 -p 11627 -i ~/.ssh/runpod


MODEL_TYPE=llama_405b QUANTIZATION=8bit poetry run uvicorn alignment_faking_mechanistic.model_server:app --host 0.0.0.0 --port 8000

curl -X POST http://0.0.0.0:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["The assistant has decided to"], "max_new_tokens": 30, "do_sample": false, "add_special_tokens": false}' | python -m json.tool

curl http://0.0.0.0:8000/health | python -m json.tool
poetry run python scripts/run_patchscope_api.py data/patchscope/config_405b.yaml

MODEL_TYPE=llama_1b poetry run uvicorn alignment_faking_mechanistic.model_server:app --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/health | python -m json.tool

