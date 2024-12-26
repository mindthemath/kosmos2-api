split:
	ffmpeg -i IMG_2867.MOV -vf "fps=30" frames/frame_%06d.png

requirements.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra viz -o requirements.txt

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api -o requirements.api.txt

build:
	docker build -t kosmos2-api .

test:
	curl -X POST -F "content=@IMG_0395.jpg" http://127.0.0.1:8000/predict | jq '.output'

lint:
	uv run black .
	uv run isort --profile black .