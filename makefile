dev:
	uv run --isolated --extra api server.py

split:
	ffmpeg -i IMG_2867.MOV -vf "fps=30" frames/frame_%06d.png

stitch:
	ffmpeg -framerate 30 -i "out/frame_%06d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p output.mp4

movie:
	uv run --isolated --extra viz movie.py

requirements.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api --extra viz -o requirements.txt

requirements.api.txt: pyproject.toml
	uv pip compile pyproject.toml --extra api -o requirements.api.txt

build: requirements.api.txt
	docker build -t kosmos2-api .

test:
	curl -X POST -F "content=@IMG_0395.jpg" http://127.0.0.1:8020/predict | jq '.output'

lint:
	uv run black .
	uv run isort --profile black .

run:
	docker run --rm -ti --gpus all -p 8020:8020 kosmos2-api:latest

tag: build
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:$$(date +%Y%m%d)-cu12.2.2
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:$$(date +%Y%m%d)
	docker tag kosmos2-api:latest mindthemath/kosmos2-api:latest
	docker images | grep mindthemath/kosmos2-api

push: tag
	docker push mindthemath/kosmos2-api:$$(date +%Y%m%d)-cu12.2.2
	docker push mindthemath/kosmos2-api:$$(date +%Y%m%d)
	docker push mindthemath/kosmos2-api:latest
