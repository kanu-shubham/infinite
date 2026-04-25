.PHONY: install server-baseline server-apc server-apc-chunked \
        bench-baseline bench-apc bench-apc-chunked compare clean

install:
	pip install -r requirements.txt

server-baseline:
	bash scripts/run_server.sh configs/baseline.env

server-apc:
	bash scripts/run_server.sh configs/apc.env

server-apc-chunked:
	bash scripts/run_server.sh configs/apc_chunked.env

bench-baseline:
	bash scripts/run_bench.sh baseline

bench-apc:
	bash scripts/run_bench.sh apc

bench-apc-chunked:
	bash scripts/run_bench.sh apc_chunked

compare:
	@echo "== shared_system: baseline -> +APC =="
	python -m bench.compare \
	  --baseline bench/results/baseline-shared.json \
	  --candidate bench/results/apc-shared.json
	@echo
	@echo "== mixed: +APC -> +APC +chunked =="
	python -m bench.compare \
	  --baseline bench/results/apc-mixed.json \
	  --candidate bench/results/apc_chunked-mixed.json

clean:
	rm -rf bench/results __pycache__ */__pycache__
