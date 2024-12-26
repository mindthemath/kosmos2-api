dev:
	uv run --isolated --extra api server.py

split:
	ffmpeg -i IMG_2867.MOV -vf "fps=30" frames/frame_%06d.png

movie:
	uv run --isolated --extra viz movie.py

requirements.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra viz -o requirements.txt

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api -o requirements.api.txt

build: requirements.api.txt
	docker build -t kosmos2-api .

test:
	curl -X POST -F "content=@IMG_0395.jpg" http://127.0.0.1:8002/predict | jq '.output'

lint:
	uv run black .
	uv run isort --profile black .

run:
	docker run --rm -ti --gpus all -p 8002:8000 kosmos2-api:latest
